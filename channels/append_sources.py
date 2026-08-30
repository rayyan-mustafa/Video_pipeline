import sys
import os

sys.path.insert(0, r"f:\wierd_bio\channels")
from pipeline.config import TOPICS

def append_sources():
    out_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\7af41bf1-8eb0-49d3-9703-1515d34a02c0\pipeline_prompts.md"
    
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n---\n\n## 5. THE TOPIC BANK (With Sources)\n\n")
        f.write("Here are all the topics in the bank along with their sources. If you are running the prompts manually, you'll need to fetch the text from these URLs to feed into the Evidence Extractor.\n\n")
        
        for key, data in TOPICS.items():
            f.write(f"### {data['title']}\n")
            f.write(f"- **Stake:** {data.get('stake', 'N/A')}\n")
            f.write("- **Sources:**\n")
            for src in data.get('sources', []):
                f.write(f"  - [{src['title']}]({src['url']})\n")
            f.write("\n")

if __name__ == "__main__":
    append_sources()
