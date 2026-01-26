
import json
import logging
import re
from typing import List, Callable, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .interfaces import ILogger, IEventBus, ICredentialStore
from .types import *
from .utils import new_id

logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
    def put(self, doc: Document) -> None:
        self._docs[doc.id] = doc
    def get(self, id: str) -> Document:
        return self._docs[id]
    def search(self, query: str) -> List[Document]:
        q = query.lower()
        return [d for d in self._docs.values() if q in d.title.lower() or q in d.content.lower()]

class StateStore:
    def __init__(self) -> None:
        self._s: dict[str, dict] = {}
    def put(self, key: str, value: dict) -> None:
        self._s[key] = value
    def get(self, key: str) -> dict:
        return self._s.get(key, {})
    def del_(self, key: str) -> None:
        self._s.pop(key, None)

class ToolService:
    def __init__(self, cred_store: ICredentialStore | None = None):
        self._tools: dict[str, ToolDef] = {}
        self._cred = cred_store
    def registerTool(self, tool: ToolDef) -> ToolRef:
        self._tools[tool.name] = tool
        return ToolRef(name=tool.name)
    def listTools(self) -> List[ToolRef]:
        return [ToolRef(name=n) for n in self._tools.keys()]
    def invokeTool(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            raise ValueError('tool not found')
        return ToolResult(name=name, output={'echo': args})

class RAGService:
    def __init__(self, store: DocumentStore):
        self.store = store
    def indexDocs(self, docs: List[Document]) -> IndexSummary:
        for d in docs:
            self.store.put(d)
        return IndexSummary(count=len(docs))
    def queryIndex(self, query: str, filters: dict) -> ContextBundle:
        return ContextBundle(documents=self.store.search(query))
    def vectorize(self, text: str) -> List[float]:
        return [float(len(text))]

class LLMService:
    def __init__(self, toolsvc: ToolService):
        self.toolsvc = toolsvc
    def generate(self, prompt: str, tools: List[ToolRef]) -> LLMOutput:
        return LLMOutput(text=f"LLM says: {prompt}")
    def stream(self, prompt: str, onToken: Callable[[str], None]) -> None:
        for t in prompt.split(' '):
            onToken(t)
    def toolCall(self, name: str, args: dict) -> ToolResult:
        return self.toolsvc.invokeTool(name, args)

class TemplateRepository:
    def getAgentTemplate(self, name: str) -> AgentTemplate:
        return AgentTemplate(name=name)
    def getWorkflowTemplate(self, name: str) -> Workflow:
        return Workflow(id=new_id('wf_'), name=name)

class LangGraphBuilder:
    def buildFromPrompt(self, prompt: str, template: AgentTemplate):
        return {'graph': 'built', 'prompt': prompt, 'template': template.name}
    def compile(self, graph) -> dict:
        return {'runnable': True}

class AgentService:
    """
    Enhanced AgentService with template matching and update detection.

    When createFromPrompt is called, the service:
    1. Uses LLM to analyze the user's intent (purpose, domain, keywords).
    2. Compares intent against predefined templates via keyword overlap.
    3. If a template matches, builds the agent from that template with user overrides.
    4. If no template matches, falls back to full LLM-powered generation via AgentDesigner.
    5. Registers the resulting agent in the AgentRegistry.

    Update Mode:
    - Detects modification keywords in prompt
    - Preserves agent name and ID when updating
    - Only modifies specified fields
    """

    # Minimum keyword overlap score to consider a template a match
    MATCH_THRESHOLD = 0.25  # Lowered for better fuzzy matching

    # Minimum similarity score to consider an existing agent a match
    AGENT_SIMILARITY_THRESHOLD = 0.25  # Lowered - we use smarter matching now

    # Keywords that indicate user wants to modify rather than create
    UPDATE_KEYWORDS = [
        "change", "modify", "update", "adjust", "edit", "alter",
        "add tool", "remove tool", "refine", "improve", "tweak",
        "fix", "enhance", "revise", "amend"
    ]

    # Tool selection rules: keyword patterns -> tool IDs
    # IMPORTANT: Keys must match actual backend tool IDs in apps/storage/tools/
    # This is synced with agent_designer.py TOOL_SELECTION_RULES
    TOOL_SELECTION_RULES = {
        "tool_web_search": [
            # Research & Information
            "research", "analyze", "analysis", "search", "web", "explore", "investigate",
            "find", "lookup", "browse", "internet", "online", "query", "discover",
            "information", "info", "data", "facts", "knowledge", "learn", "learning",
            # News & Updates
            "news", "trends", "trending", "latest", "current", "today", "recent", "update", "updates",
            # Finance & Business
            "financial", "finance", "stock", "stocks", "market", "markets", "trading", "invest", "investment",
            "report", "reports", "analyst", "advisor", "business", "company", "companies", "startup",
            "crypto", "bitcoin", "cryptocurrency", "forex", "currency", "exchange",
            # Travel & Booking
            "travel", "trip", "trips", "vacation", "holiday", "booking", "book", "reserve",
            "flight", "flights", "airline", "airlines", "airport",
            "hotel", "hotels", "accommodation", "stay", "airbnb", "hostel",
            "reservation", "reservations", "destination", "destinations",
            "tour", "tours", "tourism", "tourist", "sightseeing",
            "itinerary", "ticket", "tickets", "pass", "visa",
            "train", "trains", "railway", "bus", "buses", "car", "cars", "rental", "rentals",
            "transport", "transportation", "commute", "route", "routes", "directions",
            # Shopping & Products
            "shop", "shopping", "buy", "purchase", "price", "prices", "cost", "compare", "comparison",
            "product", "products", "item", "items", "store", "stores", "amazon", "ebay", "deal", "deals",
            "review", "reviews", "rating", "ratings", "recommendation", "recommendations",
            # Food & Dining
            "restaurant", "restaurants", "food", "foods", "menu", "dining", "eat", "eating",
            "recipe", "recipes", "cook", "cooking", "cuisine", "delivery", "takeout",
            # Entertainment
            "movie", "movies", "film", "films", "show", "shows", "tv", "television", "netflix", "stream",
            "music", "song", "songs", "artist", "artists", "album", "concert", "concerts",
            "event", "events", "sports", "game", "games", "match", "score", "scores",
            "video", "videos", "youtube", "watch",
            # Location & Maps
            "location", "locations", "address", "map", "maps", "place", "places", "nearby", "local",
            "weather", "forecast", "temperature", "climate",
            # Health & Medical
            "health", "medical", "doctor", "doctors", "hospital", "hospitals", "clinic",
            "medicine", "medication", "symptom", "symptoms", "treatment", "pharmacy",
            # Education
            "education", "school", "schools", "university", "college", "course", "courses",
            "tutorial", "tutorials", "guide", "guides", "how to", "howto", "learn",
            # Jobs & Career
            "job", "jobs", "career", "careers", "employment", "hire", "hiring", "salary", "salaries",
            "resume", "interview", "work", "remote", "freelance",
            # Real Estate
            "real estate", "property", "properties", "house", "houses", "apartment", "apartments",
            "rent", "renting", "lease", "mortgage", "realtor",
            # Social & Communication
            "social media", "twitter", "facebook", "instagram", "linkedin", "tiktok",
            "people", "person", "contact", "email", "phone",
            # Reference & Knowledge
            "wikipedia", "definition", "meaning", "translate", "translation", "language",
            "book", "books", "author", "article", "articles", "blog", "publication"
        ],
        "tool_file_reader": [
            # Document Types
            "file", "files", "document", "documents", "doc", "docs",
            "pdf", "pdfs", "word", "docx", "txt", "text",
            "markdown", "md", "html", "htm", "rtf",
            # Data Files
            "csv", "excel", "xlsx", "xls", "spreadsheet", "spreadsheets",
            "json", "xml", "yaml", "yml", "ini", "config", "configuration",
            # Actions
            "read", "reading", "parse", "parsing", "extract", "extraction",
            "load", "loading", "import", "importing", "open", "opening",
            "analyze", "scan", "scanning",
            # Content Types
            "content", "contents", "data", "table", "tables", "sheet", "sheets",
            "report", "reports", "invoice", "invoices", "receipt", "receipts",
            "contract", "contracts", "agreement", "agreements",
            "resume", "cv", "letter", "letters", "form", "forms",
            # File Operations
            "attachment", "attachments", "upload", "uploaded", "download",
            "log", "logs", "readme", "license", "changelog",
            # Media (text extraction)
            "image", "images", "photo", "photos", "screenshot", "screenshots",
            "scanned", "ocr", "transcript", "transcription"
        ],
        "tool_code_generator": [
            # Languages
            "code", "coding", "program", "programming", "script", "scripting",
            "python", "javascript", "typescript", "java", "csharp", "c#",
            "cpp", "c++", "golang", "go", "rust", "ruby", "php", "swift", "kotlin",
            "scala", "perl", "bash", "shell", "powershell", "sql",
            "html", "css", "react", "angular", "vue", "node", "nodejs",
            # Actions
            "develop", "developer", "development", "build", "building",
            "create", "creating", "generate", "generating", "write", "writing",
            "implement", "implementation", "execute", "execution", "run", "running",
            "compile", "compiling", "debug", "debugging", "fix", "fixing",
            "refactor", "refactoring", "optimize", "optimization",
            # Concepts
            "software", "application", "app", "apps", "api", "apis",
            "backend", "frontend", "fullstack", "full-stack",
            "function", "functions", "method", "methods", "class", "classes",
            "algorithm", "algorithms", "logic", "module", "modules",
            "library", "libraries", "framework", "frameworks", "sdk",
            "package", "packages", "dependency", "dependencies",
            # Testing
            "test", "tests", "testing", "unittest", "unit test", "pytest", "jest",
            "selenium", "automation", "automate", "automated",
            # Web & API
            "rest", "restful", "graphql", "websocket", "http", "https",
            "endpoint", "endpoints", "route", "routes", "request", "response",
            "authentication", "authorization", "oauth", "jwt", "token",
            # Database
            "database", "databases", "db", "query", "queries",
            "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite",
            # DevOps & Cloud
            "docker", "kubernetes", "k8s", "container", "containers",
            "aws", "azure", "gcp", "cloud", "serverless", "lambda",
            "deploy", "deployment", "ci", "cd", "pipeline", "jenkins", "github actions",
            "git", "github", "gitlab", "bitbucket", "version control",
            # Data & AI
            "scrape", "scraping", "crawler", "crawling", "bot", "bots",
            "parse", "parser", "parsing", "regex", "regular expression",
            "machine learning", "ml", "ai", "artificial intelligence",
            "data science", "pandas", "numpy", "tensorflow", "pytorch"
        ],
        "tool_code_reviewer": [
            # Core Review
            "review", "reviewer", "reviewing", "code review", "peer review",
            "check", "checking", "inspect", "inspection", "examine", "audit", "auditing",
            # Quality
            "quality", "code quality", "clean code", "best practices", "standards",
            "maintainability", "readability", "documentation", "comments",
            "style", "convention", "conventions", "formatting", "lint", "linting", "linter",
            # Analysis
            "static analysis", "analyze", "analysis", "complexity", "metrics",
            "coverage", "test coverage", "code coverage",
            # Issues
            "bug", "bugs", "issue", "issues", "error", "errors", "problem", "problems",
            "smell", "code smell", "anti-pattern", "antipattern",
            "vulnerability", "vulnerabilities", "flaw", "flaws",
            # Security
            "security", "secure", "insecure", "vulnerability", "exploit",
            "injection", "sql injection", "xss", "csrf", "sanitization", "validation",
            "encryption", "authentication", "authorization",
            # Improvement
            "improve", "improvement", "optimize", "optimization", "refactor", "refactoring",
            "suggestion", "suggestions", "feedback", "recommend", "recommendation",
            # Principles
            "solid", "dry", "kiss", "yagni", "separation of concerns",
            "design pattern", "design patterns", "architecture",
            "technical debt", "legacy", "deprecated", "upgrade", "modernize"
        ],
        "tool_calculator": [
            # Basic Math
            "calculate", "calculation", "calculations", "calculator",
            "math", "mathematics", "mathematical", "compute", "computation",
            "add", "addition", "subtract", "subtraction", "multiply", "multiplication",
            "divide", "division", "sum", "total", "difference", "product", "quotient",
            # Statistics
            "average", "mean", "median", "mode", "statistics", "statistical",
            "percentage", "percent", "ratio", "proportion", "fraction", "decimal",
            "probability", "random", "distribution", "variance", "deviation",
            "min", "minimum", "max", "maximum", "range", "count",
            # Financial
            "financial", "finance", "money", "budget", "budgeting",
            "accounting", "accountant", "bookkeeping",
            "interest", "compound interest", "simple interest", "apr", "apy",
            "loan", "loans", "mortgage", "payment", "payments", "amortization",
            "tax", "taxes", "taxation", "income", "expense", "expenses",
            "profit", "loss", "margin", "markup", "discount",
            "roi", "return", "investment", "yield", "dividend",
            "depreciation", "inflation", "gdp", "growth rate",
            "salary", "wage", "hourly", "annual", "monthly",
            "tip", "gratuity", "split", "bill",
            # Conversions
            "convert", "conversion", "unit", "units", "measurement",
            "distance", "length", "weight", "mass", "volume", "area",
            "temperature", "celsius", "fahrenheit", "kelvin",
            "speed", "velocity", "time", "duration", "age",
            "currency", "exchange rate", "forex",
            # Advanced Math
            "equation", "equations", "formula", "formulas", "solve", "solution",
            "algebra", "geometry", "trigonometry", "calculus",
            "derivative", "integral", "limit",
            "exponent", "power", "logarithm", "log", "root", "square root", "sqrt",
            "factorial", "permutation", "combination",
            "matrix", "vector", "linear algebra",
            "graph", "plot", "chart", "function",
            # Numbers
            "number", "numbers", "numeric", "numerical", "digit", "digits",
            "integer", "float", "round", "rounding", "floor", "ceil", "ceiling",
            "absolute", "abs", "positive", "negative", "sign",
            "prime", "even", "odd", "factor", "factors", "multiple", "gcd", "lcm",
            # Body/Health Calculations
            "bmi", "body mass", "calories", "calorie", "nutrition",
            "heart rate", "pace", "distance", "steps"
        ]
    }

    # Stop words to exclude from similarity matching in agent-building context
    # These words are too generic and cause false positives
    AGENT_STOP_WORDS = {
        "agent", "agents", "assistant", "assistants", "helper", "helpers",
        "bot", "bots", "ai", "system", "systems", "tool", "tools",
        "create", "creating", "build", "building", "make", "making",
        "help", "helping", "want", "need", "please", "would", "could",
        "the", "a", "an", "for", "me", "my", "that", "will", "do", "does"
    }

    # Regex pattern to detect generic agent names like "Agent 1", "Agent 2", etc.
    # These should be skipped for similarity matching
    GENERIC_AGENT_NAME_PATTERN = r'^agent\s*\d+$'

    def __init__(
        self,
        tpl_repo: TemplateRepository,
        graph_builder: LangGraphBuilder,
        cred: ICredentialStore | None = None,
        log: ILogger | None = None,
        registry=None,
        designer=None
    ):
        self.tpl_repo = tpl_repo
        self.graph_builder = graph_builder
        self.agents: dict[str, Agent] = {}
        self.log = log
        self._registry = registry
        self._designer = designer
        self._templates_cache: Optional[List[Dict[str, Any]]] = None

    def _load_templates(self) -> List[Dict[str, Any]]:
        """
        Load predefined agent templates from agent_templates.json.
        Caches after first load.

        Returns:
            List of template dicts from the JSON file.
        """
        if self._templates_cache is not None:
            return self._templates_cache

        templates_path = (
            Path(__file__).parent.parent / "apps" / "storage" / "agent_templates.json"
        )

        if not templates_path.exists():
            logger.warning(f"Templates file not found at {templates_path}")
            self._templates_cache = []
            return self._templates_cache

        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._templates_cache = data.get("templates", [])
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load templates: {e}")
            self._templates_cache = []

        return self._templates_cache

    def _get_llm(self):
        """
        Get an LLM instance via LLMManager for intent analysis.
        Uses low temperature for deterministic extraction.
        """
        from llm_manager import LLMManager
        return LLMManager.get_llm(temperature=0.1, max_tokens=1000)

    def _analyze_intent(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to extract intent keywords from the user prompt.

        Args:
            prompt: The user's natural language agent description.

        Returns:
            Dict with keys: purpose, domain, keywords, matching_roles.
            On failure, returns a basic extraction based on the raw prompt.
        """
        system_prompt = """You are an intent analyzer for an AI agent builder platform.
Given a user's prompt describing an agent they want to create, extract structured intent information.

Return a JSON object with EXACTLY this structure:
{
  "purpose": "one-line purpose of the requested agent",
  "domain": "primary domain (e.g., research, sales, support, data, content, code, hr, finance, project)",
  "keywords": ["list", "of", "relevant", "keywords", "from", "the", "prompt"],
  "matching_roles": ["possible", "role", "titles", "that", "fit"]
}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation."""

        try:
            llm = self._get_llm()
            full_prompt = f"{system_prompt}\n\nUser prompt: {prompt}\n\nJSON:"
            response = llm.invoke(full_prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Strip markdown fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            intent = json.loads(content)

            # Validate expected structure
            if not isinstance(intent, dict):
                raise ValueError("LLM returned non-dict JSON")

            # Ensure required keys exist with defaults
            intent.setdefault("purpose", prompt[:100])
            intent.setdefault("domain", "general")
            intent.setdefault("keywords", [])
            intent.setdefault("matching_roles", [])

            return intent

        except Exception as e:
            logger.warning(f"LLM intent analysis failed, using basic extraction: {e}")
            return self._basic_intent_extraction(prompt)

    def _basic_intent_extraction(self, prompt: str) -> Dict[str, Any]:
        """
        Fallback intent extraction using simple keyword parsing.
        Used when LLM call fails.
        """
        prompt_lower = prompt.lower()

        # Domain keyword mappings
        domain_keywords = {
            "research": ["research", "analyze", "investigate", "study", "explore", "literature"],
            "support": ["support", "customer", "help", "assist", "service", "inquiry"],
            "data": ["data", "analytics", "statistics", "visualization", "dataset", "insights"],
            "content": ["content", "write", "blog", "article", "copy", "creative", "writing"],
            "code": ["code", "review", "programming", "developer", "software", "debug", "python"],
            "project": ["project", "manage", "coordinate", "timeline", "task", "planning"],
            "sales": ["sales", "lead", "prospect", "customer", "product", "revenue"],
            "hr": ["hr", "human resources", "employee", "hiring", "recruitment", "onboarding"],
            "finance": ["finance", "financial", "budget", "revenue", "accounting", "report"],
        }

        # Detect domain
        detected_domain = "general"
        max_matches = 0
        for domain, kws in domain_keywords.items():
            matches = sum(1 for kw in kws if kw in prompt_lower)
            if matches > max_matches:
                max_matches = matches
                detected_domain = domain

        # Extract keywords from prompt (simple word tokenization)
        words = [w.strip(".,!?;:()[]{}\"'") for w in prompt_lower.split()]
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
                      "and", "or", "in", "on", "at", "for", "with", "that", "this",
                      "i", "want", "need", "create", "make", "build", "me", "my"}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        return {
            "purpose": prompt[:100],
            "domain": detected_domain,
            "keywords": keywords[:15],
            "matching_roles": []
        }

    def _match_template(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compare analyzed intent against all predefined templates.

        Uses keyword overlap scoring across template name, role, and description.
        Returns the best matching template if score exceeds MATCH_THRESHOLD.

        Args:
            intent: Dict from _analyze_intent with purpose, domain, keywords, matching_roles.

        Returns:
            Matched template dict, or None if no sufficient match found.
        """
        templates = self._load_templates()
        if not templates:
            return None

        intent_keywords = set(
            kw.lower() for kw in intent.get("keywords", [])
        )
        intent_domain = intent.get("domain", "").lower()
        intent_roles = set(
            r.lower() for r in intent.get("matching_roles", [])
        )
        intent_purpose = intent.get("purpose", "").lower()

        # Add domain to keyword set for matching
        if intent_domain:
            intent_keywords.add(intent_domain)

        # Add purpose words to keyword set
        purpose_words = {
            w.strip(".,!?;:()[]{}\"'") for w in intent_purpose.split()
            if len(w) > 3
        }
        intent_keywords.update(purpose_words)

        best_template = None
        best_score = 0.0

        for template in templates:
            score = self._score_template(template, intent_keywords, intent_roles)
            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= self.MATCH_THRESHOLD and best_template is not None:
            logger.info(
                f"Template match found: '{best_template.get('name')}' "
                f"(score: {best_score:.2f})"
            )
            return best_template

        logger.info(f"No template match found (best score: {best_score:.2f})")
        return None

    def _score_template(
        self,
        template: Dict[str, Any],
        intent_keywords: set,
        intent_roles: set
    ) -> float:
        """
        Score a template against the intent keywords.

        Scoring factors:
        - Keyword overlap with template name, role, description
        - Role match bonus
        - Tool keyword overlap

        Returns:
            Float score between 0.0 and 1.0.
        """
        if not intent_keywords:
            return 0.0

        # Build template keyword set from name, role, description
        template_text = " ".join([
            template.get("name", ""),
            template.get("role", ""),
            template.get("description", ""),
        ]).lower()

        template_words = {
            w.strip(".,!?;:()[]{}\"'") for w in template_text.split()
            if len(w) > 2
        }

        # Add tool names as keywords
        for tool in template.get("tools", []):
            template_words.update(
                w.lower() for w in tool.split() if len(w) > 2
            )

        # Calculate keyword overlap (Jaccard-like)
        if not template_words:
            return 0.0

        intersection = intent_keywords & template_words
        union = intent_keywords | template_words
        keyword_score = len(intersection) / len(union) if union else 0.0

        # Role match bonus: check if template role matches any intent roles
        role_bonus = 0.0
        template_role = template.get("role", "").lower()
        if template_role:
            for intent_role in intent_roles:
                if (intent_role in template_role) or (template_role in intent_role):
                    role_bonus = 0.25
                    break

        # Name match bonus
        name_bonus = 0.0
        template_name = template.get("name", "").lower()
        for kw in intent_keywords:
            if kw in template_name:
                name_bonus = 0.15
                break

        # Combined score (capped at 1.0)
        total = min(1.0, keyword_score + role_bonus + name_bonus)
        return total

    def createFromPrompt(self, prompt: str, template: AgentTemplate) -> Agent:
        """
        Create an agent from a natural language prompt with template matching.

        Flow:
        1. Analyze intent using LLM.
        2. Attempt template match.
        3. If matched: build from template with user prompt overrides.
        4. If not matched: use AgentDesigner for full LLM generation.
        5. Register in AgentRegistry.
        6. Return the Agent.

        Args:
            prompt: User's natural language description of the desired agent.
            template: AgentTemplate with optional overrides.

        Returns:
            Fully populated Agent instance.
        """
        try:
            # Step 1: Analyze intent
            intent = self._analyze_intent(prompt)
            logger.info(f"Intent analysis: domain={intent.get('domain')}, "
                        f"keywords={intent.get('keywords', [])[:5]}")

            # Step 2: Attempt template match
            matched_template = self._match_template(intent)

            if matched_template:
                # Step 3a: Build from matched template
                agent = self._build_from_template(
                    matched_template, prompt, template, intent
                )
            else:
                # Step 3b: Fall back to LLM generation via AgentDesigner
                agent = self._build_from_llm(prompt, template)

        except Exception as e:
            logger.error(f"Agent creation failed, using minimal fallback: {e}")
            # Ultimate fallback: create a basic agent
            agent = Agent(
                id=new_id('agt_'),
                name=template.name or "Agent",
                description=prompt[:200],
                metadata={
                    "source": "fallback",
                    "created_at": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            )

        # Store internally
        self.agents[agent.id] = agent

        # Register in AgentRegistry if available
        self._register_agent(agent)

        return agent

    def _build_from_template(
        self,
        matched_template: Dict[str, Any],
        prompt: str,
        user_template: AgentTemplate,
        intent: Dict[str, Any]
    ) -> Agent:
        """
        Build an Agent from a matched predefined template, applying user overrides.

        Args:
            matched_template: The template dict from agent_templates.json.
            prompt: Original user prompt (used for description customization).
            user_template: User-provided AgentTemplate with optional overrides.
            intent: Analyzed intent dict.

        Returns:
            Fully populated Agent.
        """
        agent_id = new_id('agt_')
        timestamp = datetime.utcnow().isoformat()

        # Use user template overrides if provided, else fall back to matched template
        name = user_template.name if user_template.name else matched_template.get("name", "Agent")
        icon = user_template.icon if user_template.icon else matched_template.get("icon", "")
        role = user_template.role if user_template.role else matched_template.get("role", "Processing")
        description = (
            user_template.description if user_template.description
            else matched_template.get("description", prompt[:200])
        )
        agent_prompt = (
            user_template.prompt if user_template.prompt
            else matched_template.get("prompt", prompt)
        )

        # Tools: user override > template tools > auto-select based on intent
        tools = user_template.tools if user_template.tools else matched_template.get("tools", [])
        if not tools:
            # Auto-select tools based on prompt/intent if neither provided
            tools = self._auto_select_tools(prompt, intent)

        variables = (
            user_template.variables if user_template.variables
            else matched_template.get("variables", [])
        )
        settings = (
            user_template.settings if user_template.settings
            else matched_template.get("settings", {})
        )

        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            tools=tools,
            metadata={
                "source": "template",
                "template_name": matched_template.get("name"),
                "icon": icon,
                "prompt": agent_prompt,
                "variables": variables,
                "settings": settings,
                "original_prompt": prompt,
                "intent_domain": intent.get("domain", "general"),
                "created_by": "agent_service",
                "created_at": timestamp,
            }
        )

        return agent

    def _build_from_llm(self, prompt: str, template: AgentTemplate) -> Agent:
        """
        Build an Agent using AgentDesigner for full LLM-powered generation.

        Args:
            prompt: User's natural language description.
            template: AgentTemplate with optional overrides.

        Returns:
            Agent built from LLM-designed spec.
        """
        if self._designer is None:
            # Import and create designer if not injected
            from apps.agent.designer.agent_designer import AgentDesigner
            self._designer = AgentDesigner()

        # Use AgentDesigner to generate full agent spec
        # Pass None for tools if not provided to trigger auto-selection
        agent_dict = self._designer.design_from_prompt(
            user_prompt=prompt,
            default_model="openrouter-devstral",
            icon=template.icon or "",
            tools=template.tools if template.tools else None,
            variables=template.variables or []
        )

        # Tag as LLM-generated
        if "metadata" not in agent_dict:
            agent_dict["metadata"] = {}
        agent_dict["metadata"]["source"] = "llm_generated"
        agent_dict["metadata"]["original_prompt"] = prompt

        # Convert dict to Agent model
        agent = Agent(
            id=agent_dict.get("agent_id", new_id('agt_')),
            name=agent_dict.get("name", template.name or "Agent"),
            role=agent_dict.get("role"),
            description=agent_dict.get("description"),
            tools=agent_dict.get("tools"),
            metadata=agent_dict.get("metadata")
        )

        return agent

    def _register_agent(self, agent: Agent) -> None:
        """
        Register the agent in the AgentRegistry for persistence.

        Args:
            agent: The Agent to register.
        """
        if self._registry is None:
            return

        try:
            # Build registry-compatible dict
            agent_dict = {
                "agent_id": agent.id,
                "name": agent.name,
                "role": agent.role or "Processing",
                "description": agent.description or "",
                "icon": (agent.metadata or {}).get("icon", ""),
                "prompt": (agent.metadata or {}).get("prompt", ""),
                "tools": agent.tools or [],
                "variables": (agent.metadata or {}).get("variables", []),
                "settings": (agent.metadata or {}).get("settings", {}),
                "input_schema": agent.input_schema or [],
                "output_schema": agent.output_schema or [],
                "metadata": agent.metadata or {},
            }
            self._registry.register_agent(agent_dict)
            logger.info(f"Agent {agent.id} registered in registry")
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")

    def _auto_select_tools(self, prompt: str, intent: Dict[str, Any] = None) -> list:
        """
        Auto-select appropriate tools based on prompt and intent keywords.

        Uses keyword matching to determine which tools are relevant for the agent.
        Maximum of 2 tools will be selected.

        Args:
            prompt: The user's natural language prompt
            intent: Optional pre-analyzed intent dict with keywords

        Returns:
            List of tool IDs (max 2), empty list if no clear match
        """
        prompt_lower = prompt.lower()

        keywords = set()
        # Add words from prompt
        words = [w.strip(".,!?;:()[]{}\"'") for w in prompt_lower.split()]
        keywords.update(w for w in words if len(w) > 2)

        # Add keywords from intent if available
        if intent:
            keywords.update(kw.lower() for kw in intent.get("keywords", []))
            domain = intent.get("domain", "")
            if domain:
                keywords.add(domain.lower())

        if not keywords:
            return []

        # Score each tool based on keyword matches
        tool_scores = {}

        for tool_id, tool_keywords in self.TOOL_SELECTION_RULES.items():
            score = 0
            for keyword in tool_keywords:
                if keyword in prompt_lower:
                    score += 2  # Higher score for direct prompt match
                elif keyword in keywords:
                    score += 1

            if score > 0:
                tool_scores[tool_id] = score

        if not tool_scores:
            return []

        # Sort by score and take top 2
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool_id for tool_id, score in sorted_tools[:2]]

        return selected_tools

    def _detect_update_intent(self, prompt: str) -> bool:
        """
        Detect if the user's prompt indicates modification of an existing agent
        rather than creation of a new one.

        Args:
            prompt: User's natural language prompt

        Returns:
            True if prompt indicates update/modification intent
        """
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.UPDATE_KEYWORDS)

    def classify_user_intent(
        self,
        context: str,
        suggested_value: str,
        user_message: str,
        conversation_history: list = None
    ) -> Dict[str, Any]:
        """
        Classify user intent using LLM reasoning.

        This method enables natural language understanding for conversational flows.
        Instead of pattern matching, it uses LLM to understand what the user means.

        Args:
            context: The conversation context (e.g., "name_confirmation", "refinement")
            suggested_value: The value being confirmed/modified (e.g., agent name)
            user_message: The user's natural language response
            conversation_history: Optional list of previous messages

        Returns:
            Dict with:
            - intent: CONFIRMATION | MODIFICATION | REJECTION | CLARIFICATION
            - confidence: float between 0 and 1
            - reasoning: explanation of the classification
            - extracted_value: new value if intent is MODIFICATION
        """
        system_prompt = """You are an intent classifier for an AI agent builder conversation.

Your task is to analyze the user's message and classify their intent.

CONTEXT TYPES:
- name_confirmation: User is responding to "Would you like to use this name?"
- refinement: User is responding to "Any changes or are we ready to finalize?"
- tool_selection: User is responding to tool configuration
- general: General conversation

INTENT TYPES:
1. CONFIRMATION - User is accepting/approving the suggested value
   Examples: "yes", "ok", "keep this name", "sounds good", "that's perfect", "love it", "this name is great"

2. MODIFICATION - User wants to change/replace the value
   Examples: "call it X instead", "change to Y", "I prefer Z", "rename it to..."

3. REJECTION - User is declining/refusing
   Examples: "no", "I don't like it", "start over", "cancel"

4. CLARIFICATION - User is asking a question or needs more info
   Examples: "what does this mean?", "can you explain?", "what options do I have?"

CRITICAL RULES:
- If user expresses ANY form of approval, agreement, satisfaction, or acceptance -> CONFIRMATION
- Natural language variations like "oh yes this is great", "keep this name", "this name is the best" -> CONFIRMATION
- Only classify as MODIFICATION if user EXPLICITLY provides a new value or asks to change
- When in doubt between CONFIRMATION and MODIFICATION, prefer CONFIRMATION if no new value is provided

Return JSON only:
{
  "intent": "CONFIRMATION" | "MODIFICATION" | "REJECTION" | "CLARIFICATION",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "extracted_value": null or "the new value if MODIFICATION"
}"""

        user_prompt = f"""Context: {context}
Suggested value: "{suggested_value}"
User message: "{user_message}"

Classify the user's intent. Return JSON only."""

        try:
            llm = self._get_llm()
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.invoke(full_prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Strip markdown fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Validate and ensure required fields
            valid_intents = ["CONFIRMATION", "MODIFICATION", "REJECTION", "CLARIFICATION"]
            if result.get("intent") not in valid_intents:
                result["intent"] = "CONFIRMATION"  # Safe default

            result.setdefault("confidence", 0.8)
            result.setdefault("reasoning", "Intent classified by LLM")
            result.setdefault("extracted_value", None)

            logger.info(f"Intent classified: {result['intent']} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            # Fallback: simple heuristic
            return self._fallback_intent_classification(user_message, suggested_value)

    def _fallback_intent_classification(self, user_message: str, suggested_value: str) -> Dict[str, Any]:
        """
        Fallback intent classification when LLM is unavailable.
        Uses simple heuristics as last resort.
        """
        msg_lower = user_message.lower().strip()

        # Check for obvious rejections
        rejection_words = ["no", "nope", "cancel", "stop", "quit", "nevermind", "never mind"]
        if any(msg_lower == word or msg_lower.startswith(word + " ") for word in rejection_words):
            return {
                "intent": "REJECTION",
                "confidence": 0.7,
                "reasoning": "Fallback: detected rejection keyword",
                "extracted_value": None
            }

        # Check for questions
        if "?" in user_message or any(msg_lower.startswith(q) for q in ["what", "how", "why", "can", "could", "would"]):
            return {
                "intent": "CLARIFICATION",
                "confidence": 0.6,
                "reasoning": "Fallback: detected question pattern",
                "extracted_value": None
            }

        # Check for modification indicators
        modification_patterns = ["call it", "name it", "rename", "change to", "change it to", "prefer", "instead"]
        for pattern in modification_patterns:
            if pattern in msg_lower:
                # Try to extract new value
                extracted = user_message
                for p in modification_patterns:
                    if p in msg_lower:
                        parts = msg_lower.split(p)
                        if len(parts) > 1:
                            extracted = parts[1].strip().strip("\"'")
                            break
                return {
                    "intent": "MODIFICATION",
                    "confidence": 0.7,
                    "reasoning": "Fallback: detected modification pattern",
                    "extracted_value": extracted if extracted != user_message else None
                }

        # Default: treat as confirmation (most common case)
        return {
            "intent": "CONFIRMATION",
            "confidence": 0.6,
            "reasoning": "Fallback: no rejection/modification detected, assuming confirmation",
            "extracted_value": None
        }

    def updateFromPrompt(
        self,
        agent_id: str,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Update an existing agent based on user prompt.

        This method preserves the agent's name and ID while only updating
        the fields specified in the user's prompt.

        Args:
            agent_id: ID of the agent to update
            prompt: User's natural language update request

        Returns:
            Dict with action type and updated agent definition

        Raises:
            ValueError: If agent not found
        """
        if self._registry is None:
            raise ValueError("Registry not available")

        # Get existing agent
        existing_agent = self._registry.get_agent(agent_id)
        if not existing_agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        # Use designer to generate updates while preserving identity
        if self._designer is None:
            from apps.agent.designer.agent_designer import AgentDesigner
            self._designer = AgentDesigner()

        # Call update_from_prompt which preserves name and ID
        updated_agent = self._designer.update_from_prompt(
            existing_agent=existing_agent,
            user_prompt=prompt
        )

        # Save updated agent to registry
        self._registry.update_agent(agent_id, updated_agent)

        logger.info(f"Agent {agent_id} updated via prompt")

        return {
            "action": "UPDATE_AGENT",
            "agent_id": agent_id,
            "agent_name": updated_agent.get("name"),
            "agent": updated_agent
        }

    def _normalize_word(self, word: str) -> str:
        """
        Simple word normalization to handle common variations.
        Maps related words to a common root for better matching.
        """
        word = word.lower().strip(".,!?;:()[]{}\"'")

        # Common word family mappings
        normalizations = {
            # Analysis family
            "analyst": "analy", "analysis": "analy", "analyze": "analy",
            "analyzes": "analy", "analyzing": "analy", "analytical": "analy",
            # Finance family
            "financial": "financ", "finance": "financ", "finances": "financ",
            # Research family
            "research": "research", "researcher": "research", "researching": "research",
            # Code family
            "code": "code", "coding": "code", "coder": "code", "coded": "code",
            # Data family
            "data": "data", "dataset": "data", "datasets": "data",
            # Support family
            "support": "support", "supporting": "support", "supporter": "support",
            # Content family
            "content": "content", "contents": "content",
            # Write family
            "write": "write", "writer": "write", "writing": "write", "written": "write",
            # Project family
            "project": "project", "projects": "project",
            # Manage family
            "manage": "manag", "manager": "manag", "management": "manag", "managing": "manag",
            # Sales family
            "sales": "sale", "sale": "sale", "selling": "sale",
            # Customer family
            "customer": "custom", "customers": "custom",
            # Review family
            "review": "review", "reviewer": "review", "reviewing": "review", "reviews": "review",
            # HR family
            "hr": "hr", "human": "human", "resources": "resource", "resource": "resource",
        }

        return normalizations.get(word, word)

    def _score_template_match(self, template: Dict[str, Any], user_prompt: str, intent_keywords: set) -> float:
        """
        Score how well a template matches the user's prompt using multiple strategies.

        Scoring:
        1. Direct name match in prompt: HIGH score (0.8+)
        2. All name words found in prompt: HIGH score (0.7+)
        3. Normalized word overlap: MEDIUM score
        4. Jaccard similarity: baseline

        Returns score between 0.0 and 1.0
        """
        template_name = template.get("name", "").lower()
        template_role = template.get("role", "").lower()
        user_prompt_lower = user_prompt.lower()

        # PRE-CHECK: Skip generic agent names like "Agent 1", "Agent 2", "Agent 3"
        # These should not be used for similarity matching as they're placeholder names
        if re.match(self.GENERIC_AGENT_NAME_PATTERN, template_name, re.IGNORECASE):
            logger.debug(f"Skipping generic agent name: '{template_name}'")
            return 0.0

        # Strategy 1: Direct name match (template name appears in prompt)
        # e.g., "financial analyst" in "create a financial analyst agent"
        # But skip if the name is just a stop word
        if template_name and template_name in user_prompt_lower:
            # Verify it's not just matching stop words
            name_significant_words = [w for w in template_name.split()
                                      if w not in self.AGENT_STOP_WORDS and len(w) > 2]
            if len(name_significant_words) >= 1:
                return 0.95  # Near-perfect match

        # Strategy 2: All significant words from template name appear in prompt
        # IMPORTANT: Filter out stop words to avoid false positives like "Agent 3" matching any prompt with "agent"
        name_words = [w.strip() for w in template_name.split()
                      if len(w) > 2 and w.strip() not in self.AGENT_STOP_WORDS]

        if name_words:
            # Check if all name words (or their normalized forms) appear in prompt
            # Also filter stop words from prompt to avoid matching on common words
            prompt_words = {w for w in user_prompt_lower.split()
                          if w not in self.AGENT_STOP_WORDS}
            prompt_normalized = {self._normalize_word(w) for w in prompt_words}

            name_matches = 0
            for nw in name_words:
                nw_normalized = self._normalize_word(nw)
                # Check exact match or normalized match
                if nw in prompt_words or nw_normalized in prompt_normalized:
                    name_matches += 1
                # Check partial match (word contains or is contained)
                elif any(nw in pw or pw in nw for pw in prompt_words if len(pw) > 3):
                    name_matches += 0.7

            # Require at least 2 significant words for high-confidence match
            # This prevents single-word matches from getting 0.85 score
            if len(name_words) >= 2:
                name_match_ratio = name_matches / len(name_words)
                if name_match_ratio >= 0.8:
                    return 0.85
                elif name_match_ratio >= 0.5:
                    return 0.6 + (name_match_ratio * 0.2)
            elif len(name_words) == 1 and name_matches >= 1:
                # Single significant word match gets lower score (needs role/description confirmation)
                # This prevents "Agent 3" (after filtering) from matching everything
                return 0.4  # Lower score for single-word matches

        # Strategy 3: Role match (also filter stop words)
        if template_role:
            role_words = [w.strip() for w in template_role.split()
                         if len(w) > 2 and w.strip() not in self.AGENT_STOP_WORDS]
            prompt_words_filtered = {w for w in user_prompt_lower.split()
                                    if w not in self.AGENT_STOP_WORDS}
            role_matches = sum(1 for rw in role_words if rw in prompt_words_filtered or
                             self._normalize_word(rw) in {self._normalize_word(w) for w in prompt_words_filtered})
            if role_words and role_matches / len(role_words) >= 0.5:
                return 0.5 + (role_matches / len(role_words) * 0.3)

        # Strategy 4: Normalized keyword overlap (fallback, also filter stop words)
        template_text = f"{template_name} {template_role} {template.get('description', '')}".lower()
        template_words = {self._normalize_word(w) for w in template_text.split()
                        if len(w) > 2 and w not in self.AGENT_STOP_WORDS}
        intent_normalized = {self._normalize_word(kw) for kw in intent_keywords
                           if kw.lower() not in self.AGENT_STOP_WORDS}

        if template_words and intent_normalized:
            intersection = template_words & intent_normalized
            union = template_words | intent_normalized
            jaccard = len(intersection) / len(union) if union else 0.0
            return jaccard * 0.8  # Scale Jaccard to max 0.8

        return 0.0

    def _check_existing_agents(self, intent: Dict[str, Any], user_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar agent already exists in templates or registry.

        Uses smart matching that handles word variations (analyst/analysis/analyze).
        First checks predefined templates, then the registry.

        Args:
            intent: Analyzed intent dict with keywords, domain, purpose
            user_prompt: Original user prompt for direct matching

        Returns:
            Dict with matching agent/template info if found, None otherwise
        """
        # Build intent keyword set
        intent_keywords = set(kw.lower() for kw in intent.get("keywords", []))
        intent_domain = intent.get("domain", "").lower()
        intent_purpose = intent.get("purpose", "").lower()

        # Use purpose as prompt if not provided
        if not user_prompt:
            user_prompt = intent_purpose

        # Add domain and purpose words to keywords
        if intent_domain:
            intent_keywords.add(intent_domain)
        purpose_words = {
            w.strip(".,!?;:()[]{}\"'") for w in intent_purpose.split()
            if len(w) > 3
        }
        intent_keywords.update(purpose_words)

        if not intent_keywords and not user_prompt:
            return None

        # --- Step 1: Check templates FIRST ---
        templates = self._load_templates()
        best_template = None
        best_template_score = 0.0

        for template in templates:
            score = self._score_template_match(template, user_prompt, intent_keywords)

            if score > best_template_score:
                best_template_score = score
                best_template = template

        # Match threshold: 0.4 for smart matching (lower than before since scoring is smarter)
        if best_template_score >= 0.4 and best_template is not None:
            logger.info(
                f"Template match found: '{best_template.get('name')}' "
                f"(score: {best_template_score:.2f})"
            )
            # Auto-select tools based on template purpose to get proper tool IDs
            template_text = f"{best_template.get('name', '')} {best_template.get('role', '')} {best_template.get('description', '')}"
            auto_tools = self._auto_select_tools(template_text, None)

            # Create a copy of the template with auto-selected tool IDs
            template_with_tools = dict(best_template)
            template_with_tools["tools"] = auto_tools

            # Return template as pseudo-agent with correct tool IDs
            return {
                "action": "AGENT_EXISTS",
                "agent_id": best_template.get("id", f"tpl_{best_template.get('name', 'unknown').lower().replace(' ', '_')}"),
                "agent_name": best_template.get("name"),
                "similarity_score": round(best_template_score, 2),
                "message": "A similar agent template already exists. You can configure or modify it.",
                "agent": template_with_tools,
                "source": "template"
            }

        # --- Step 2: Check registry ---
        if self._registry is None:
            return None

        existing_agents = self._registry.list_agents()
        if not existing_agents:
            return None

        best_match = None
        best_score = 0.0

        for agent in existing_agents:
            # Build a pseudo-template from agent for scoring
            agent_as_template = {
                "name": agent.get("name", ""),
                "role": agent.get("role", ""),
                "description": agent.get("description", "")
            }
            score = self._score_template_match(agent_as_template, user_prompt, intent_keywords)

            if score > best_score:
                best_score = score
                best_match = agent

        if best_score >= 0.4 and best_match is not None:
            logger.info(
                f"Similar agent found: '{best_match.get('name')}' "
                f"(similarity: {best_score:.2f})"
            )
            return {
                "action": "AGENT_EXISTS",
                "agent_id": best_match.get("agent_id"),
                "agent_name": best_match.get("name"),
                "similarity_score": round(best_score, 2),
                "message": "A similar agent already exists. You can configure or modify it.",
                "agent": best_match,
                "source": "registry"
            }

        return None

    def createFromCanvasCard(self, cardJSON: dict, template: AgentTemplate) -> Agent:
        """
        Build a proper Agent from canvas card JSON (template data).

        Maps all template fields into the Agent structure and applies
        user-provided overrides from the template parameter.

        Args:
            cardJSON: Dict with template/card data (name, icon, role, description,
                      prompt, tools, variables, settings).
            template: AgentTemplate with optional overrides.

        Returns:
            Fully populated Agent.
        """
        agent_id = new_id('agt_')
        timestamp = datetime.utcnow().isoformat()

        # Resolve each field: user override > card data > default
        name = template.name if template.name else cardJSON.get("name", "Agent")
        icon = template.icon if template.icon else cardJSON.get("icon", "")
        role = template.role if template.role else cardJSON.get("role", "Processing")
        description = (
            template.description if template.description
            else cardJSON.get("description", "")
        )
        agent_prompt = (
            template.prompt if template.prompt
            else cardJSON.get("prompt", "")
        )
        tools = (
            template.tools if template.tools
            else cardJSON.get("tools", [])
        )
        variables = (
            template.variables if template.variables
            else cardJSON.get("variables", [])
        )
        settings = (
            template.settings if template.settings
            else cardJSON.get("settings", {})
        )

        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            tools=tools,
            metadata={
                "source": cardJSON.get("source", "canvas_card"),
                "icon": icon,
                "prompt": agent_prompt,
                "variables": variables,
                "settings": settings,
                "created_by": "agent_service",
                "created_at": timestamp,
            }
        )

        # Store and register
        self.agents[agent.id] = agent
        self._register_agent(agent)

        return agent

    def validateA2A(self, agent: Agent) -> ValidationResult:
        """Validate agent-to-agent communication compatibility."""
        return ValidationResult(ok=True)

    def listAgents(self) -> List[Agent]:
        """List all agents created by this service instance."""
        return list(self.agents.values())

class WorkflowService:
    def __init__(self, agentsvc: AgentService, bus: IEventBus | None = None):
        self.agentsvc = agentsvc
        self.bus = bus
    def createFromPrompt(self, prompt: str, agents: List[Agent]) -> Workflow:
        wf = Workflow(id=new_id('wf_'), name='wf_from_prompt')
        return wf
    def createFromCanvas(self, canvasJSON: dict) -> Workflow:
        return Workflow(id=new_id('wf_'), name='wf_from_canvas')
    def validate(self, workflow: Workflow) -> ValidationResult:
        return ValidationResult(ok=True)
    def publish(self, workflow: Workflow) -> None:
        pass

class ConnectorManager:
    def __init__(self, mcp, custom):
        self.mcp = mcp
        self.custom = custom
        self._conns: dict[str, ConnectorDef] = {}
    def register(self, conn: ConnectorDef) -> ConnectorRef:
        self._conns[conn.name] = conn
        return ConnectorRef(name=conn.name)
    def invoke(self, name: str, payload: dict) -> ConnectorResult:
        if name not in self._conns:
            raise ValueError('connector not found')
        return ConnectorResult(name=name, result={'payload': payload})
    def list(self) -> List[ConnectorRef]:
        return [ConnectorRef(name=n) for n in self._conns.keys()]

class MCPConnector:
    def connect(self, endpoint: str, token: str):
        return {'connected': True}
    def sendMessage(self, sessionId: str, content: dict) -> dict:
        return {'ok': True, 'sessionId': sessionId, 'content': content}
    def getSession(self, sessionId: str) -> dict:
        return {'sessionId': sessionId}

class CustomConnector:
    def __init__(self):
        self._cfg = {}
    def configure(self, config: dict) -> None:
        self._cfg = config
    def call(self, operation: str, payload: dict) -> ConnectorResult:
        return ConnectorResult(name=operation, result={'payload': payload})
    def healthCheck(self) -> Health:
        return Health(status='ok')

class ChatOrchestrator:
    def __init__(self, bus: IEventBus, llm: LLMService, rag: RAGService, wf: WorkflowService, ag: AgentService, conns: ConnectorManager, log: ILogger | None = None):
        self.bus = bus
        self.llm = llm
        self.rag = rag
        self.wf = wf
        self.ag = ag
        self.conns = conns
        self.log = log
    def orchestrate(self, msg: dict, session: Session) -> dict:
        ctx = self.rag.queryIndex(msg.get('content',''), {})
        out = self.llm.generate(msg.get('content',''), [])
        self.bus.publish('chat.events', Event(type='chat.completion', data={'text': out.text}))
        return {'reply': out.text, 'ctx_docs': [d.id for d in ctx.documents]}
    def routeToAgent(self, msg: dict) -> Agent:
        return Agent(id='agt_default', name='default')
    def routeToService(self, intent: str):
        return {'service': intent}
    def publishEvent(self, topic: str, event: Event) -> None:
        self.bus.publish(topic, event)
