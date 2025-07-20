import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

colors = {
    'data_source': '#FFB6C1',
    'processing': '#FFA07A',
    'embedding': '#FF6347',
    'storage': '#DC143C',
    'frontend': '#90EE90',
    'backend': '#FFD700',
    'query': '#DDA0DD',
    'retrieval': '#20B2AA',
    'llm': '#9370DB',
    'response': '#8A2BE2',
    'evaluation': '#FF69B4'
}

def create_box(x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return box

create_box(0.5, 10.5, 2, 0.8, "PDF Books\n(PDF Files)", colors['data_source'], 9)
create_box(0.5, 9.5, 2, 0.8, "Science Books\n(CSV Data)", colors['data_source'], 9)
create_box(0.5, 8.5, 2, 0.8, "English Books\n(CSV Data)", colors['data_source'], 9)

create_box(3.5, 10, 2, 0.8, "PDF Text\nExtraction", colors['processing'], 9)
create_box(3.5, 9, 2, 0.8, "Text Chunking\n(Recursive Splitter)", colors['processing'], 9)
create_box(3.5, 8, 2, 0.8, "Document\nProcessing", colors['processing'], 9)

create_box(6.5, 9.5, 2, 0.8, "Embedding\nGeneration", colors['embedding'], 9)
ax.text(7.5, 8.8, "Ollama + Nomic\nText Embedding", ha='center', va='center',
        fontsize=8, style='italic', color='gray')

create_box(9.5, 9.5, 2, 0.8, "Vector Storage\n(ChromaDB)", colors['storage'], 9)

create_box(12.5, 10.5, 2.5, 0.8, "Frontend:\nStreamlit UI", colors['frontend'], 9)
create_box(12.5, 9.5, 2.5, 0.8, "Backend:\nFastAPI", colors['backend'], 9)

create_box(6.5, 7, 2, 0.8, "User Query\nProcessing", colors['query'], 9)

create_box(6.5, 5.5, 2, 0.8, "Contextual\nCompression\nRetriever", colors['retrieval'], 8)

create_box(9.5, 5.5, 2, 0.8, "Language Model\n(Groq + LLaMA 3)", colors['llm'], 9)
ax.text(10.5, 4.8, "LLaMA 3 8B\nvia Groq API", ha='center', va='center',
        fontsize=8, style='italic', color='gray')

create_box(12.5, 5.5, 2.5, 0.8, "Response\nGeneration", colors['response'], 9)

create_box(0.5, 6.5, 2, 0.8, "Evaluation\nDataset", colors['evaluation'], 9)
create_box(0.5, 5.5, 2, 0.8, "Metrics\nCalculation", colors['evaluation'], 9)
create_box(0.5, 4.5, 2, 0.8, "Performance\nAnalysis", colors['evaluation'], 9)

create_box(9.5, 7.5, 2, 0.8, "Vector DB\n(Physics)", colors['storage'], 9)
create_box(12.5, 7.5, 2.5, 0.8, "Vector DB\n(Science/English)", colors['storage'], 9)

def add_arrow(start_x, start_y, end_x, end_y, color='black', style='->', linewidth=1.5):
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y),
                          "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc=color, ec=color, linewidth=linewidth)
    ax.add_patch(arrow)

add_arrow(2.5, 10.9, 3.5, 10.4)
add_arrow(2.5, 9.9, 3.5, 9.4)
add_arrow(2.5, 8.9, 3.5, 8.4)

add_arrow(5.5, 10.4, 6.5, 9.9)
add_arrow(5.5, 9.4, 6.5, 9.9)
add_arrow(5.5, 8.4, 6.5, 9.9)

add_arrow(8.5, 9.9, 9.5, 9.9)

add_arrow(10.5, 9.1, 10.5, 8.3)
add_arrow(10.5, 9.1, 13.75, 8.3)

add_arrow(12.5, 9.9, 8.5, 7.4)

add_arrow(7.5, 7.4, 7.5, 6.3)

add_arrow(8.5, 5.9, 9.5, 5.9)

add_arrow(11.5, 5.9, 12.5, 5.9)

add_arrow(13.75, 5.9, 13.75, 9.9)

add_arrow(10.5, 7.1, 7.5, 6.3)
add_arrow(13.75, 7.1, 7.5, 6.3)

add_arrow(2.5, 6.9, 6.5, 7.4)
add_arrow(2.5, 5.9, 6.5, 5.9)
add_arrow(2.5, 4.9, 6.5, 5.9)

ax.text(8, 3.5, "Key Technologies:", fontsize=12, fontweight='bold', ha='center')
ax.text(8, 3, "• Ollama (Embeddings)", fontsize=10, ha='center')
ax.text(8, 2.7, "• ChromaDB (Vector Storage)", fontsize=10, ha='center')
ax.text(8, 2.4, "• Groq API (LLM)", fontsize=10, ha='center')
ax.text(8, 2.1, "• FastAPI (Backend)", fontsize=10, ha='center')
ax.text(8, 1.8, "• Streamlit (Frontend)", fontsize=10, ha='center')

ax.text(2, 11.5, "Data Ingestion", fontsize=12, fontweight='bold', ha='center')
ax.text(6, 11.5, "Processing", fontsize=12, fontweight='bold', ha='center')
ax.text(10, 11.5, "Storage", fontsize=12, fontweight='bold', ha='center')
ax.text(14, 11.5, "Interface", fontsize=12, fontweight='bold', ha='center')

ax.text(2, 3.5, "Evaluation", fontsize=12, fontweight='bold', ha='center')

ax.text(8, 12.5, "RAG-Enhanced Document Tutor System Architecture",
        fontsize=16, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('rag_enhanced_document_tutor_architecture.png', dpi=300, bbox_inches='tight')
plt.show()