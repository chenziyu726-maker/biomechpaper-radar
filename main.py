import json
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import yaml
from openai import OpenAI

SENT_FILE = "sent_pmids.json"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sent_pmids():
    if not os.path.exists(SENT_FILE):
        return []
    with open(SENT_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


def save_sent_pmids(pmids):
    with open(SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(pmids[-1000:], f, indent=2, ensure_ascii=False)


def search_pubmed(journals, keywords, days_back, max_results):
    journal_query = " OR ".join([f'"{j}"[Journal]' for j in journals])
    keyword_query = " OR ".join([f'"{k}"[Title/Abstract]' for k in keywords])
    full_query = f"({journal_query}) AND ({keyword_query})"

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": full_query,
        "reldate": days_back,
        "datetype": "edat",
        "retmax": max_results,
        "retmode": "json",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_details(pmids):
    if not pmids:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else "无标题"
        journal = (
            article.findtext(".//Journal/ISOAbbreviation")
            or article.findtext(".//Journal/Title", default="未知期刊")
        )

        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label", "")
            text = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        if not abstract:
            continue

        doi = ""
        pmcid = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            id_type = aid.attrib.get("IdType")
            if id_type == "doi" and aid.text:
                doi = aid.text.strip()
            elif id_type == "pmc" and aid.text:
                pmcid = aid.text.strip()

        link = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "abstract": abstract,
                "doi": doi,
                "pmcid": pmcid,
                "link": link,
            }
        )

    return articles


def get_free_pdf_link(doi, pmcid, email):
    """通过 Unpaywall API 和 PMC 自动查找合法免费全文链接"""
    if doi:
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            resp = requests.get(url, params={"email": email}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                best_oa = data.get("best_oa_location") or {}
                pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
                if pdf_url:
                    return pdf_url
        except Exception:
            pass

    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

    return ""


def call_llm_fast(client, model, prompt):
    """优先关闭深度思考模式以加快速度；若接口不支持该参数则自动回退标准调用"""
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err_msg = str(e).lower()
        if any(k in err_msg for k in ["enable_thinking", "extra", "invalid", "400", "unknown"]):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        raise


def process_single_article(client, model, article, email):
    prompt = f"""你是一名运动生物力学与运动康复领域的资深科研助手。请阅读以下英文文献摘要，输出一份结构清晰的中文深度科研拆解，总字数控制在 400~550 字左右，不要有任何开场白或结束语。

严格按照以下 5 个模块输出：
【中文标题】准确翻译英文标题
【研究痛点】用 1~2 句话说明当前临床或生物力学研究尚未解决的问题及本研究目的
【实验与方法】详细提取受试者样本量与分组、测试动作范式、测量设备/信号类型（如动捕、测力台、肌电、IMU 或视觉算法）以及统计/机器学习模型（若摘要未提及某项则如实略过，不要编造）
【量化结果】提炼核心生物力学指标变化方向，尽量保留摘要中的具体数值（如 P 值、效应量 SMD、AUC、敏感度/特异度、角度或力矩差异等）
【方法学与科研启发】用 1~2 句话点评该研究的实验设计亮点、局限性，或对康复评估与后续科研选题的参考价值

英文标题：{article['title']}
期刊：{article['journal']}
摘要：{article['abstract']}
"""
    summary = call_llm_fast(client, model, prompt)
    free_pdf = get_free_pdf_link(article["doi"], article["pmcid"], email)

    if free_pdf:
        pdf_line = f"📥 免费全文：{free_pdf}"
    else:
        pdf_line = "🔒 免费全文：暂无开源版本（可点上方 DOI 或校内库查阅）"

    formatted_msg = (
        f"📄 期刊：{article['journal']}\n"
        f"🌐 原题：{article['title']}\n\n"
        f"{summary}\n\n"
        f"🔗 原文：{article['link']}\n"
        f"{pdf_line}"
    )
    return article["pmid"], formatted_msg


def send_to_feishu_in_batches(webhook, messages, batch_size=1):
    total = len(messages)
    success_count = 0
    for i in range(0, total, batch_size):
        batch = messages[i : i + batch_size]
        end_idx = min(i + batch_size, total)
        if batch_size == 1:
            header = f"🧬 生物力学顶刊深度拆解 ({i + 1}/{total})\n\n"
        else:
            header = f"🧬 生物力学顶刊深度拆解 ({i + 1}-{end_idx}/{total})\n\n"
        content = header + "\n\n━━━━━━━━━━\n\n".join(batch)

        try:
            resp = requests.post(
                webhook,
                json={"msg_type": "text", "content": {"text": content}},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                print(f"第 {i // batch_size + 1} 批发送失败: {data}")
            else:
                success_count += len(batch)
                print(f"第 {i // batch_size + 1} 批发送成功 ({i + 1}-{end_idx}/{total})")
        except Exception as e:
            print(f"第 {i // batch_size + 1} 批请求异常: {e}")

        time.sleep(1)
    return success_count


def main():
    start_time = time.time()
    cfg = load_config()
    sent_pmids = load_sent_pmids()
    unpaywall_email = cfg.get("unpaywall_email", "biomech.radar@example.com")
    max_articles = cfg.get("max_articles", 20)
    max_workers = cfg.get("max_workers", 3)

    print("开始检索 PubMed 文献...")
    pmids = search_pubmed(
        cfg["journals"], cfg["keywords"], cfg["days_back"], max_articles * 3
    )
    new_pmids = [p for p in pmids if p not in sent_pmids][:max_articles]

    if not new_pmids:
        print("本次未检索到新的未推送文献。")
        return

    print(f"检索到 {len(new_pmids)} 篇新文献，正在获取摘要详情...")
    articles = fetch_details(new_pmids)
    if not articles:
        print("新文献均无摘要内容，跳过推送。")
        return

    api_key = os.environ["LLM_API_KEY"].strip()
    base_url = os.environ["LLM_BASE_URL"].strip()
    model = os.environ["LLM_MODEL"].strip()
    webhook = os.environ["FEISHU_WEBHOOK"].strip()

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    print(f"开始并发拆解 {len(articles)} 篇文献（并发数：{max_workers}，已尝试关闭深度思考）...")
    results_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_art = {
            executor.submit(process_single_article, client, model, art, unpaywall_email): art
            for art in articles
        }
        for future in as_completed(future_to_art):
            art = future_to_art[future]
            try:
                pmid, msg = future.result()
                results_map[pmid] = msg
                print(f"完成拆解 PMID: {pmid} ({len(results_map)}/{len(articles)})")
            except Exception as e:
                print(f"拆解 PMID {art['pmid']} 失败: {e}")

    summaries = []
    pushed_pmids = []
    for art in articles:
        if art["pmid"] in results_map:
            summaries.append(results_map[art["pmid"]])
            pushed_pmids.append(art["pmid"])

    if summaries:
        sent_ok = send_to_feishu_in_batches(
            webhook, summaries, cfg.get("feishu_batch_size", 1)
        )
        if sent_ok > 0:
            sent_pmids.extend(pushed_pmids)
            save_sent_pmids(sent_pmids)
            elapsed = time.time() - start_time
            print(f"全部完成：成功推送 {sent_ok} 篇文献并更新去重记录，总耗时 {elapsed:.1f} 秒。")


if __name__ == "__main__":
    main()
