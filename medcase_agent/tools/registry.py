from .pubmed_tools import search_pubmed, fetch_ama_citations
from .clingen_tools import search_clingen_by_keyword, fetch_clingen_variant_data
from .medgemma_tools import analyze_radiology_image
from .composite_tools import analyze_composite_figure
from .disease_importance_tools import assess_disease_importance

# 1. Define the schema for the LLM
PUBMED_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "Use this tool to find real, recent, peer-reviewed medical literature candidates. "
            "DO NOT HALLUCINATE OR INVENT CITATIONS. References in the final output MUST be verified "
            "using this tool and MUST include a valid DOI or PMID. "
            "Use this tool for scientific context, epidemiology, diagnostic criteria, and treatment guidelines. "
            "For manuscript generation, retrieve enough candidates to support at least 10 verified final references. "
            "Do not cite every returned result; select a compact set of high-relevance sources and avoid citation stuffing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The medical search query. Use strict Boolean keyword matching (e.g., 'sarcoidosis AND arrhythmias'). "
                        "Do not use natural language sentences."
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return. Default is 10; use 10 to 15 when building a manuscript citation bank.",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

FETCH_CITATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_ama_citations", # Updated to plural to match the new function name
        "description": (
            "Given a selected list of DOIs (Digital Object Identifiers) obtained from the search_pubmed tool, "
            "this tool returns formatted, sequentially numbered AMA (American Medical Association) citation strings. "
            "You can pass a single DOI or multiple DOIs at once. "
            "Use this tool to build the References section from only the sources that will be cited inline. "
            "For manuscript generation, request no fewer than 10 selected DOIs when available so the final article can meet the minimum citation requirement. "
            "The formatted citation strings returned by this tool are the only approved source for final References entries; do not let the LLM invent reference strings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dois": { # Changed from 'doi' to 'dois'
                    "type": "array", # Changed type to array so it can accept a list
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of exact DOIs for the papers (e.g., ['10.1001/jama.2023.1', '10.1056/NEJMoa123']). Do not include URLs."
                }
            },
            "required": ["dois"]
        }
    }
}

DISEASE_IMPORTANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "assess_disease_importance",
        "description": (
            "Retrieve similar prior cases from the local disease index to support a grounded "
            "literature and novelty plan. Source and related articles are excluded by the runtime. "
            "Retrieval matches are evidence candidates, not disease classifications or proof of novelty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "diseases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Disease names to search. Supply these explicitly, or provide related_keywords. "
                        "The tool does not infer queries from case metadata."
                    )
                },
                "related_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional presentation or differential keywords that help refine retrieval, such as the unusual feature, organ, or mimic."
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum candidate references returned after source and related-article exclusions. Default is 8.",
                    "default": 8
                }
            }
        }
    }
}

CLINGEN_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_clingen_by_keyword",
        "description": (
            "STEP 1 OF GENETIC SEARCH: Use this tool when the patient data mentions a specific genetic variant or mutation "
            "(e.g., 'RPGR c.1512_1513del' or 'BRAF V600E'). This tool searches the NCBI database to find the official ClinGen UUID. "
            "You MUST use this tool first to get the UUID before you can fetch the detailed genetic evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The gene and variant name. e.g., 'RPGR c.1512_1513del'. Do not include extra text."
                }
            },
            "required": ["query"]
        }
    }
}

CLINGEN_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_clingen_variant_data",
        "description": (
            "STEP 2 OF GENETIC SEARCH: Takes a ClinGen UUID (obtained from the search_clingen_by_keyword tool) and returns "
            "expert-curated medical evidence, disease condition, and pathogenicity classification. Use this data to write "
            "the 'Genetic Findings' section of the case report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "uuid": {
                    "type": "string",
                    "description": "The exact UUID string returned by the search_clingen_by_keyword tool."
                }
            },
            "required": ["uuid"]
        }
    }
}

MEDGEMMA_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_radiology_image",
        "description": (
            "Analyzes specialized medical images to identify notable clinical findings. "
            "CRITICAL USAGE GUIDELINES: "
            "1. MODALITY LIMIT: Only use this tool for CT, MRI, X-ray, and histology images. Do not use for general charts or photos. "
            "2. NATIVE VISION FIRST: You have your own vision capabilities. Use this tool carefully as a supplemental expert opinion. "
            "3. COMPOSITE IMAGE WARNING: This tool attempts to auto-split composite figures (e.g., panels A, B, C). However, "
            "the returned panels (e.g., 'panel_1', 'panel_2') are sorted by a basic layout algorithm and may NOT perfectly map to the original A/B/C labels. "
            "4. CLUTTERED LAYOUTS: If the sub-figures in a composite image are aligned extremely close together without clear whitespace, "
            "the splitting tool will fail. In those cases, rely entirely on your own native vision to analyze the image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_reference_id": {
                    "type": "string",
                    "description": "The virtual Reference ID of the image exactly as provided in the prompt (e.g., 'IMG_A1B2C3')."
                },
                "query": {
                    "type": "string",
                    "description": "Specific question about the image. Leave default for a general clinical analysis.",
                    "default": "Can you analyze this medical image and describe any notable clinical findings?"
                }
            },
            "required": ["image_reference_id"]
        }
    }
}

COMPOSITE_FIGURE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_composite_figure",
        "description": (
            "Analyzes complex, multi-panel composite medical images (e.g., figures containing sub-labels like A, B, C, "
            "or mixed imaging modalities). "
            "CRITICAL USAGE GUIDELINES: "
            "1. Use this tool ONLY when the image contains multiple sub-figures or panels. "
            "2. It automatically pulls clinical context to generate a comprehensive, panel-by-panel description. "
            "3. Preserve visible panel labels exactly (e.g., A and B) so the manuscript can mention Figure 1A and Figure 1B in the main text. "
            "4. Use standard `analyze_radiology_image` for single-modality, single-frame scans."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_reference_id": {
                    "type": "string",
                    "description": "The virtual Reference ID of the composite image exactly as provided in the prompt (e.g., 'IMG_A1B2C3')."
                },
                "query": {
                    "type": "string",
                    "description": "Specific focus areas, structural requirements, or formatting instructions for the vision model (e.g., 'Explicitly separate analysis into Panel A and Panel B, and keep caption-relevant findings concise')."
                }
            },
            "required": ["image_reference_id"]
        }
    }
}

# ==========================================
# TOOL REGISTRY AND MAPPING
# ==========================================

AVAILABLE_TOOLS = {
    "search_pubmed": search_pubmed,
    "search_clingen_by_keyword": search_clingen_by_keyword,
    "fetch_clingen_variant_data": fetch_clingen_variant_data,
    "analyze_radiology_image": analyze_radiology_image,
    "fetch_ama_citations": fetch_ama_citations,
    "analyze_composite_figure": analyze_composite_figure,
    "assess_disease_importance": assess_disease_importance,
}

TOOL_SCHEMAS = [
    PUBMED_TOOL_SCHEMA,
    DISEASE_IMPORTANCE_SCHEMA,
    CLINGEN_SEARCH_SCHEMA,
    CLINGEN_FETCH_SCHEMA,
    MEDGEMMA_TOOL_SCHEMA,
    FETCH_CITATION_SCHEMA,
    COMPOSITE_FIGURE_SCHEMA,
]
