import json
import os
import time
import xml.etree.ElementTree as ET
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
    # 只保留最近 1000 条记录，防止文件无限变大
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

        # 提取完整摘要（兼容带分段标签和斜体标签的结构）
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label", "")
            text = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        # 跳过没有摘要的评论或勘误文章
        if not abstract:
            continue

        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        link = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "abstract": abstract,
                "link": link,
            }
        )

    return articles


def summarize_article(client, model, article):
    prompt = f"""你是一名运动生物力学与运动康复领域的研究助手。请阅读以下英文文献摘要，用简洁的中文提炼核心信息，严格控制在 150 字以内，不要有多余开场白。

输出格式固定为：
【中文标题】...
【对象与方法】一句话说明受试者、动作范式或设备/算法
【核心发现】一句话说明关键生物力学指标变化与临床/科研意义

英文标题：{article['title']}
期刊：{article['journal']}
摘要：{article['abstract']}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    summary = resp.choices[0].message.content.strip()
    return f"📄 {article['journal']}\n{summary}\n🔗 原文：{article['link']}"


def send_to_feishu_in_batches(webhook, messages, batch_size=2):
    total = len(messages)
    for i in range(0, total, batch_size):
        batch = messages[i : i + batch_size]
        end_idx = min(i + batch_size, total)
        header = f"🧬 生物力学顶刊追踪 ({i + 1}-{end_idx}/{total})\n\n"
        content = header + "\n\n━━━━━━━━━━\n\n".join(batch)

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
            print(f"第 {i // batch_size + 1} 批发送成功 ({i + 1}-{end_idx}/{total})")

        # 间隔 1 秒，避免触发飞书频率限制
        time.sleep(1)


def main():
    cfg = load_config()
    sent_pmids = load_sent_pmids()

    print("开始检索 PubMed 文献...")
    pmids = search_pubmed(
        cfg["journals"], cfg["keywords"], cfg["days_back"], cfg["max_articles"]
    )
    new_pmids = [p for p in pmids if p not in sent_pmids]

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

    client = OpenAI(api_key=api_key, base_url=base_url)

    summaries = []
    pushed_pmids = []
    for idx, art in enumerate(articles, 1):
        try:
            print(f"正在总结第 {idx}/{len(articles)} 篇 (PMID: {art['pmid']})...")
            summaries.append(summarize_article(client, model, art))
            pushed_pmids.append(art["pmid"])
            time.sleep(0.5)
        except Exception as e:
            print(f"总结 PMID {art['pmid']} 失败: {e}")

    if summaries:
        send_to_feishu_in_batches(webhook, summaries, cfg.get("feishu_batch_size", 2))
        sent_pmids.extend(pushed_pmids)
        save_sent_pmids(sent_pmids)
        print(f"全部完成：成功推送 {len(summaries)} 篇文献并更新去重记录。")


if __name__ == "__main__":
    main()
