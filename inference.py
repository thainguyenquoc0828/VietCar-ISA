import os
import re
import json
import asyncio
import argparse
import aiohttp
import nest_asyncio
import pandas as pd
from tqdm.asyncio import tqdm

SYSTEM_PROMPT = """
# SYSTEM ROLE
You are a sentiment analysis expert for Vietnamese car comments. Your task is to detect the implicit sentiment towards the car in each sentence. Analyse the sentence step by step to explain the implicit meaning, then conclude with the final sentiment: Positive or Negative. Use JSON output format only.

# OUTPUT FORMAT
{
  "analysis": Brief explanation,
  "sentiment": Positive or Negative
}

# FEW-SHOT EXAMPLES
Example 1:
Q: Đổ đầy bình từ hôm mùng 1, ngày nào cũng chạy đi làm cả đi cả về 30km qua đoạn Trường Chinh tắc đường mà nay 25 rồi kim xăng mới báo quá nửa chút.
A: {
"analysis": "Việc chạy xe liên tục hàng ngày trong điều kiện tắc đường nhưng mức hao hụt xăng rất ít (chạy 25 ngày mới hết nửa bình) ngầm phản ánh động cơ xe tối ưu và cực kỳ tiết kiệm nhiên liệu",
"sentiment": "Positive"
}

Example 2:
Q: Đừng đi xe này nếu bạn không muốn cảm giác dính ghế!
A: {
"analysis": "Dùng cấu trúc phủ định để khen ngầm sức mạnh động cơ và khả năng tăng tốc rất bốc (dính ghế)",
"sentiment": "Positive"
}

Example 3:
Q: Cái cửa gió chắc hãng lắp vào cho có, ngồi đằng sau tưởng như đang đi xông hơi
A: {
  "analysis": "Dùng hình ảnh 'đi xông hơi' để chê bai trực diện hiệu năng làm mát của hệ thống điều hòa quá yếu",
  "sentiment": "Negative"
}

Example 4:
Q: Bỏ gần hai tỷ mua con xe này liệu có đáng không cả nhà?
A: {
"analysis": "Dùng câu hỏi tu từ để ngầm thể hiện sự hoài nghi và cho rằng chất lượng chiếc xe không hề xứng đáng với mức giá cao",
"sentiment": "Negative"
}

Example 5:
Q: Chán nhất là đi mấy con xe hãng này, chạy chục vạn km rồi mà nó chả chịu hỏng để lấy cớ xin vợ đổi xe mới.
A: {
  "analysis": "Dùng từ ngữ tiêu cực 'chán nhất' để mở đầu nhưng thực chất là 'than phiền' việc xe quá bền bỉ, không hề bị lỗi hỏng vặt",
  "sentiment": "Positive"
}

Example 6:
Q: Với mức giá này, bạn nên giữ cho cả nhà bạn dùng. Anh em chúng tôi dùng tạm Vios là được rồi!
A: {
"analysis": "Dùng lời khuyên mỉa mai và hạ mình đi xe bình dân (Vios) để chê bai chiếc xe bị định giá quá cao, hoang tưởng",
"sentiment": "Negative"
}

Now analyze the following sentence. Provide your step-by-step reasoning first, and then conclude with the JSON format

Q: {text}
A:
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation on VietCar-ISA dataset")
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--input_path", type=str, default="data/VietCar-ISA.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--api_url", type=str, default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=10000)
    return parser.parse_args()

def extract_text_from_response(data):
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output")
    if isinstance(output, list):
        texts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content_list = item.get("content")
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                elif isinstance(text, dict):
                    value = text.get("value")
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
        if texts:
            return "\n".join(texts)

    return None

def remove_markdown_code_fence(text):
    text = str(text).strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

def extract_json_object(text):
    cleaned_text = remove_markdown_code_fence(text)

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned_text[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in the output")

def parse_model_output(raw_output):
    if raw_output is None:
        raise ValueError("The response does not contain any output content")

    parsed = extract_json_object(raw_output)
    analysis = parsed.get("analysis")
    sentiment = parsed.get("sentiment")

    if isinstance(analysis, list):
        analysis = " ".join(str(item).strip() for item in analysis if str(item).strip())

    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError("The 'analysis' field is missing or invalid")

    if not isinstance(sentiment, str):
        raise ValueError("The 'sentiment' field is missing or invalid")

    normalized_sentiment = sentiment.strip().lower()
    if normalized_sentiment == "positive":
        normalized_sentiment = "Positive"
    elif normalized_sentiment == "negative":
        normalized_sentiment = "Negative"
    else:
        raise ValueError(f"Invalid sentiment value: {sentiment!r}")

    return {
        "analysis": analysis.strip(),
        "sentiment": normalized_sentiment
    }

async def read_response_data(response):
    response_text = await response.text()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_response": response_text}

async def call_api(session, comment, semaphore, args, api_key):
    async with semaphore:
        user_content = f"Q: {comment}\nA:"
        last_raw_output = None
        last_error = None

        for attempt in range(1, args.max_retries + 1):
            try:
                payload = {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0,
                    "max_tokens": args.max_tokens
                }

                async with session.post(
                    args.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                ) as response:
                    data = await read_response_data(response)

                    if response.status != 200:
                        last_error = f"HTTP {response.status}: {json.dumps(data, ensure_ascii=False)}"
                        if attempt < args.max_retries:
                            await asyncio.sleep(2 ** min(attempt, 5))
                            continue
                        return f"ERROR: {last_error}", f"ERROR: {last_error}"

                    raw_output = extract_text_from_response(data)
                    last_raw_output = raw_output
                    parsed_output = parse_model_output(raw_output)

                    return parsed_output["analysis"], parsed_output["sentiment"]

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)}"
                if attempt < args.max_retries:
                    await asyncio.sleep(2 ** min(attempt, 5))
                    continue

        raw_text = last_raw_output if last_raw_output is not None else "No model output"
        error_msg = f"ERROR: {last_error}"
        return raw_text, error_msg

async def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found at: {args.input_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_safe_name = args.model.replace("/", "_").replace("-", "_")
    output_path = os.path.join(args.output_dir, f"{model_safe_name}_output.csv")

    df = pd.read_csv(args.input_path)
    if "Text" not in df.columns:
        raise ValueError("The 'Text' column was not found in the dataset")

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for comment in df["Text"]:
            if pd.isna(comment) or not str(comment).strip():
                tasks.append(asyncio.sleep(0, result=("SKIP", "SKIP")))
            else:
                tasks.append(
                    call_api(
                        session=session,
                        comment=str(comment).strip(),
                        semaphore=semaphore,
                        args=args,
                        api_key=api_key
                    )
                )

        results = await tqdm.gather(*tasks, desc=f"Inference: {args.model}")

    df["explanation"] = [item[0] for item in results]
    df["predicted_sentiment"] = [item[1] for item in results]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())