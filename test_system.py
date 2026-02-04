#!/usr/bin/env python3
"""
Test script to verify the NCP-GENL study system functionality
Tests: Content loading, paper tracking, data structures
"""

import json
import os
from datetime import datetime

# Import core modules (avoiding streamlit import)
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flashcard_content_enhanced import FLASHCARD_CONTENT_ENHANCED, get_flashcard_content

# Mock DOMAINS and FLASHCARD_TOPICS from app.py
DOMAINS = {
    "Model Optimization": {"weight": 17, "target": 75},
    "GPU Acceleration & Distributed Training": {"weight": 14, "target": 75},
    "Prompt Engineering": {"weight": 13, "target": 75},
    "Fine-Tuning": {"weight": 13, "target": 75},
    "Data Preparation": {"weight": 9, "target": 75},
    "Model Deployment": {"weight": 9, "target": 75},
    "Evaluation": {"weight": 7, "target": 75},
    "Production Monitoring & Reliability": {"weight": 7, "target": 75},
    "LLM Architecture": {"weight": 6, "target": 75},
    "Safety, Ethics & Compliance": {"weight": 5, "target": 75}
}

FLASHCARD_TOPICS = {
    "LLM Architecture": [
        "Multi-Head Attention", "Feed-Forward Blocks", "Residual Connections",
        "Layer Normalization", "Self-Attention (Q, K, V)", "KV Cache",
        "Absolute Positional Encoding", "Rotary Positional Encoding (RoPE)",
        "Decoder-Only Architecture", "Encoder-Decoder Architecture", "Scaling Laws"
    ],
    "Prompt Engineering": [
        "Zero-Shot", "One-Shot", "Few-Shot", "Chain-of-Thought (CoT)",
        "Self-Consistency", "System Message", "User Message", "Tool/Function Messages",
        "JSON-Only Output", "Delimiters", "Content Filters", "Temperature",
        "Domain Adaptation via Prompts", "RAG Prompting"
    ],
    "Data Preparation": [
        "Data Collection", "Data Cleaning", "De-duplication", "Filtering",
        "BPE (Byte Pair Encoding)", "WordPiece", "Vocab Size", "Special Tokens",
        "Pre-train Split", "Fine-tune Split", "Eval/Test Split", "Leakage",
        "RAG Chunking", "Overlap", "Metadata"
    ],
    "Model Optimization": [
        "FP32", "FP16", "INT8", "INT4", "Weight Quantization",
        "Activation Quantization", "PTQ", "QAT", "TensorRT-LLM Graph Fusion",
        "Kernel Auto-Tuning", "Pruning", "Distillation", "Beam Search",
        "Sampling", "Temperature", "Top-k Sampling", "Top-p Sampling", "KV Cache Optimization"
    ],
    "Fine-Tuning": [
        "Full Fine-Tune", "LoRA", "QLoRA", "Rank (r)", "Alpha",
        "Target Modules", "Learning Rate", "Warmup", "Batch Size",
        "Epochs", "Early Stopping", "Instruction Tuning", "Domain Adaptation",
        "Safety Tuning", "Catastrophic Forgetting", "Data Mixing"
    ],
    "GPU Acceleration & Distributed Training": [
        "Tensor Cores", "Mixed Precision", "Batch Size vs VRAM",
        "Gradient Accumulation", "Data Parallelism", "Tensor Parallelism",
        "Pipeline Parallelism", "Gradient Checkpointing", "Offloading",
        "ZeRO", "NCCL", "All-Reduce", "Communication Overhead", "Scaling Efficiency"
    ],
    "Model Deployment": [
        "Triton Model Repository", "Model Config (config.pbtxt)", "Dynamic Batching",
        "Concurrent Models", "HTTP REST/gRPC", "NIM Packaging", "NIM Routing",
        "NIM Scaling", "Docker", "GPU Runtime", "Blue-Green", "Canary", "Shadow"
    ],
    "Evaluation": [
        "Perplexity", "Log-Loss", "ROUGE-L", "ROUGE-1", "ROUGE-2",
        "BLEU", "BERTScore", "Accuracy/F1", "Human Evaluation Rubrics",
        "Pairwise Comparison", "Test Harnesses", "A/B Testing"
    ],
    "Production Monitoring & Reliability": [
        "Latency (P50, P95, P99)", "Throughput", "Error Rate", "Timeout Rate",
        "Cache Hit Rate", "SLO", "SLI", "Error Budget", "Data Drift",
        "Concept Drift", "Model Drift", "Alerting", "Rollback", "Capacity Planning"
    ],
    "Safety, Ethics & Compliance": [
        "Pre-Prompt Filters", "Post-Generation Filters", "Blocklists", "Allowlists",
        "Violence Content", "Hate Speech", "PII Detection", "PII Redaction",
        "Bias Detection", "Toxicity Mitigation", "GDPR", "Auditability"
    ]
}

DATA_FILE = "study_data.json"

def test_content_loading():
    """Test flashcard content loading"""
    print("=" * 60)
    print("TEST 1: Content Loading")
    print("=" * 60)
    
    # Test getting content for various topics
    test_cases = [
        ("LLM Architecture", "Multi-Head Attention"),
        ("Prompt Engineering", "Chain-of-Thought (CoT)"),
        ("Model Optimization", "LoRA"),
        ("Fine-Tuning", "LoRA"),
        ("Evaluation", "Perplexity"),
        ("Safety, Ethics & Compliance", "PII Detection")
    ]
    
    passed = 0
    failed = 0
    
    for domain, topic in test_cases:
        content = get_flashcard_content(domain, topic)
        if content and isinstance(content, dict):
            has_definition = "definition" in content
            has_papers = len([l for l in content.get("links", []) if l.get("type") == "paper"]) > 0
            if has_definition and has_papers:
                print(f"✅ {domain} > {topic}")
                print(f"   Papers: {len([l for l in content.get('links', []) if l.get('type') == 'paper'])}")
                passed += 1
            else:
                print(f"⚠️  {domain} > {topic} - Missing definition or papers")
                failed += 1
        else:
            print(f"❌ {domain} > {topic} - No content found")
            failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed\n")
    return passed, failed

def test_coverage():
    """Test content coverage"""
    print("=" * 60)
    print("TEST 2: Content Coverage")
    print("=" * 60)
    
    total_topics = sum(len(topics) for topics in FLASHCARD_TOPICS.values())
    enhanced_topics = sum(
        1 for domain, topics in FLASHCARD_TOPICS.items()
        for topic in topics
        if get_flashcard_content(domain, topic) is not None
    )
    
    print(f"Total topics defined: {total_topics}")
    print(f"Topics with enhanced content: {enhanced_topics}")
    print(f"Coverage: {enhanced_topics/total_topics*100:.1f}%\n")
    
    # Coverage by domain
    print("Coverage by Domain:")
    for domain, topics in FLASHCARD_TOPICS.items():
        enhanced = sum(1 for topic in topics if get_flashcard_content(domain, topic) is not None)
        coverage_pct = enhanced / len(topics) * 100
        status = "✅" if coverage_pct >= 50 else "⚠️"
        print(f"  {status} {domain}: {enhanced}/{len(topics)} ({coverage_pct:.1f}%)")
    
    print()
    return enhanced_topics, total_topics

def test_papers():
    """Test research papers"""
    print("=" * 60)
    print("TEST 3: Research Papers")
    print("=" * 60)
    
    papers_by_domain = {}
    total_papers = 0
    
    for domain, topics in FLASHCARD_CONTENT_ENHANCED.items():
        domain_papers = 0
        for topic, content in topics.items():
            if isinstance(content, dict) and "links" in content:
                papers = [l for l in content["links"] if l.get("type") == "paper"]
                domain_papers += len(papers)
                total_papers += len(papers)
        papers_by_domain[domain] = domain_papers
    
    print(f"Total research papers: {total_papers}\n")
    print("Papers by Domain:")
    for domain, count in sorted(papers_by_domain.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} papers")
    
    print()
    return total_papers

def test_data_structure():
    """Test data file structure"""
    print("=" * 60)
    print("TEST 4: Data Structure")
    print("=" * 60)
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Data file exists: {DATA_FILE}")
        print(f"   Flashcards: {sum(len(domain) for domain in data.get('flashcards', {}).values())} cards")
        print(f"   Papers: {len(data.get('papers', {}))} papers tracked")
        print(f"   Domains: {len(data.get('domain_mastery', {}))} domains")
        
        # Check if papers structure is correct
        papers = data.get("papers", {})
        if papers:
            sample_paper = list(papers.values())[0]
            required_fields = ["title", "url", "domain", "topic", "read"]
            missing_fields = [f for f in required_fields if f not in sample_paper]
            if missing_fields:
                print(f"⚠️  Missing fields in papers: {missing_fields}")
            else:
                print(f"✅ Paper structure correct")
        else:
            print("ℹ️  No papers tracked yet (will be initialized when viewing flashcards)")
        
        print()
        return True
    else:
        print(f"ℹ️  Data file doesn't exist yet (will be created on first run)")
        print()
        return False

def test_enhanced_content_structure():
    """Test enhanced content structure"""
    print("=" * 60)
    print("TEST 5: Enhanced Content Structure")
    print("=" * 60)
    
    required_fields = ["definition", "links"]
    optional_fields = ["formula", "diagram", "code_example"]
    
    sample_topics = []
    for domain, topics in list(FLASHCARD_CONTENT_ENHANCED.items())[:3]:
        for topic, content in list(topics.items())[:2]:
            if isinstance(content, dict):
                sample_topics.append((domain, topic, content))
    
    print(f"Testing {len(sample_topics)} sample topics:\n")
    
    passed = 0
    for domain, topic, content in sample_topics:
        has_required = all(field in content for field in required_fields)
        has_optional = any(field in content for field in optional_fields)
        has_papers = len([l for l in content.get("links", []) if l.get("type") == "paper"]) > 0
        
        if has_required and has_papers:
            status = "✅"
            passed += 1
        else:
            status = "❌"
        
        print(f"{status} {domain} > {topic}")
        print(f"   Definition: {'✅' if 'definition' in content else '❌'}")
        print(f"   Papers: {len([l for l in content.get('links', []) if l.get('type') == 'paper'])}")
        print(f"   Formula: {'✅' if 'formula' in content else '○'}")
        print(f"   Diagram: {'✅' if 'diagram' in content else '○'}")
        print(f"   Code: {'✅' if 'code_example' in content else '○'}")
        print()
    
    print(f"Result: {passed}/{len(sample_topics)} topics have required structure\n")
    return passed == len(sample_topics)

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NCP-GENL STUDY SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test 1: Content Loading
    passed, failed = test_content_loading()
    results["content_loading"] = {"passed": passed, "failed": failed}
    
    # Test 2: Coverage
    enhanced, total = test_coverage()
    results["coverage"] = {"enhanced": enhanced, "total": total, "percentage": enhanced/total*100}
    
    # Test 3: Papers
    total_papers = test_papers()
    results["papers"] = {"total": total_papers}
    
    # Test 4: Data Structure
    data_exists = test_data_structure()
    results["data_structure"] = {"exists": data_exists}
    
    # Test 5: Enhanced Content Structure
    structure_ok = test_enhanced_content_structure()
    results["structure"] = {"ok": structure_ok}
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Enhanced Topics: {enhanced}/{total} ({enhanced/total*100:.1f}%)")
    print(f"✅ Research Papers: {total_papers}")
    print(f"✅ Content Loading: {passed} tests passed")
    print(f"✅ Data Structure: {'OK' if data_exists else 'Will be created'}")
    print(f"✅ Content Structure: {'OK' if structure_ok else 'Issues found'}")
    print()
    
    # Overall status
    if enhanced >= 80 and total_papers >= 200 and passed >= 5:
        print("🎉 SYSTEM STATUS: EXCELLENT - Ready for use!")
    elif enhanced >= 60 and total_papers >= 150:
        print("✅ SYSTEM STATUS: GOOD - Minor improvements possible")
    else:
        print("⚠️  SYSTEM STATUS: NEEDS IMPROVEMENT")
    
    print()
    print("=" * 60)
    print("To run the Streamlit app:")
    print("  streamlit run app.py")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
