"""
Agent designer service.
Generates agent definitions from natural language prompts using LLM.

LLM Provider Configuration:
---------------------------
This module supports multiple LLM providers. Configure via .env file:
- OPTION 1: Ollama (On-Premise) - Set USE_OLLAMA=true
- OPTION 2: OpenRouter (Current) - Set USE_OPENROUTER=true
- OPTION 3: Azure OpenAI - Set USE_AZURE=true
- OPTION 4: OpenAI Direct - Set USE_OPENAI=true

See .env file for detailed configuration options.
"""
import json
import os
from typing import Dict, Any, Optional
from echolib.utils import new_id
from datetime import datetime


class AgentDesigner:
    """
    Agent designer service.
    Uses LLM to generate agent definitions from natural language prompts.
    """

    # Available tools for auto-selection (must match actual backend tool IDs)
    AVAILABLE_TOOLS = ["tool_web_search", "tool_file_reader", "tool_code_generator", "tool_code_reviewer", "tool_calculator"]

    # Tool selection rules: keyword patterns -> tool IDs
    # IMPORTANT: Keys must match actual backend tool IDs in apps/storage/tools/
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

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize designer.

        Args:
            api_key: OpenAI API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._llm_client = None
        self._llm_providers = self._load_llm_providers()

    def _load_llm_providers(self) -> Dict[str, Any]:
        """Load LLM provider configurations from llm_provider.json."""
        provider_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "llm_provider.json"
        )
        if os.path.exists(provider_file):
            with open(provider_file, 'r') as f:
                data = json.load(f)
                return {model["id"]: model for model in data.get("models", [])}
        return {}

    def _get_llm_client(self, model_id: str = None):
        """
        Get LLM client using centralized LLM Manager.

        All LLM configuration is now in llm_manager.py
        To change provider/model, edit llm_manager.py

        Args:
            model_id: Optional model ID (ignored, kept for backward compatibility)

        Returns:
            LLM client instance
        """
        from llm_manager import LLMManager

        # Get LLM from centralized manager
        # Uses default configuration from llm_manager.py
        try:
            return LLMManager.get_llm(
                temperature=0.3,
                max_tokens=4000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get LLM from LLMManager: {e}")

    def _select_tools_for_agent(self, user_prompt: str, intent: Dict[str, Any] = None) -> list:
        """
        Auto-select appropriate tools based on agent purpose and intent keywords.

        Uses keyword matching to determine which tools are relevant for the agent.
        Maximum of 2 tools will be selected to avoid over-tooling.

        Args:
            user_prompt: The user's natural language prompt
            intent: Optional pre-analyzed intent dict with keywords

        Returns:
            List of tool IDs (max 2), empty list if no clear match
        """
        # Build keyword set from prompt and intent
        prompt_lower = user_prompt.lower()

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
                # Check if keyword appears in prompt or intent keywords
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

    def design_from_prompt(
        self,
        user_prompt: str,
        default_model: str = "openrouter-devstral",
        icon: str = "",
        tools: list = None,
        variables: list = None
    ) -> Dict[str, Any]:
        """
        Design agent from user prompt using LLM.

        Args:
            user_prompt: Natural language description of agent
            default_model: Model ID to use for agent
            icon: Emoji icon for agent
            tools: List of tool names (if None or empty, auto-selects based on purpose)
            variables: List of variable definitions

        Returns:
            Agent definition dict
        """
        if variables is None:
            variables = []

        # Auto-select tools if not provided or empty
        if tools is None or len(tools) == 0:
            tools = self._select_tools_for_agent(user_prompt)

        # Use LLM to analyze prompt
        try:
            agent_spec = self._design_with_llm(user_prompt, default_model)
        except Exception as e:
            print(f"LLM design failed, using basic structure: {e}")
            agent_spec = self._design_basic(user_prompt)

        # Build agent definition
        agent_id = new_id("agt_")
        timestamp = datetime.utcnow().isoformat()

        # Get LLM-suggested settings or use defaults
        llm_settings = agent_spec.get("settings", {})
        temperature = llm_settings.get("temperature", 0.7)
        max_tokens = llm_settings.get("max_tokens", 2000)
        top_p = llm_settings.get("top_p", 0.9)
        max_iterations = llm_settings.get("max_iterations", 5)

        agent = {
            "agent_id": agent_id,
            "name": agent_spec.get("name", "Agent"),
            "icon": icon,
            "role": agent_spec.get("role", "Processing"),
            "description": agent_spec.get("description", user_prompt[:200]),
            "prompt": agent_spec.get("prompt", user_prompt),
            "model": default_model,
            "tools": tools,
            "variables": variables,
            "settings": {
                "temperature": temperature,
                "max_token": max_tokens,
                "top_p": top_p,
                "max_iteration": max_iterations
            },
            "input_schema": agent_spec.get("input_schema", []),
            "output_schema": agent_spec.get("output_schema", []),
            "constraints": {
                "max_steps": max_iterations,
                "timeout_seconds": 60
            },
            "permissions": {
                "can_call_agents": False,
                "allowed_agents": []
            },
            "metadata": {
                "created_by": "agent_designer",
                "created_at": timestamp,
                "tags": ["auto-generated"]
            }
        }

        return agent

    def _design_with_llm(
        self,
        user_prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """Design agent using LLM analysis."""

        system_prompt = """You are an AI agent designer. Analyze the user's request and design a complete agent specification.

Return a JSON response with this exact structure:
{
  "name": "Creative and Memorable Agent Name",
  "role": "Professional Role/Title",
  "description": "What this agent does (1-2 sentences)",
  "prompt": "Detailed system prompt/instructions for the agent",
  "input_schema": ["list", "of", "input", "keys"],
  "output_schema": ["list", "of", "output", "keys"],
  "settings": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "max_iterations": 5
  }
}

IMPORTANT RULES:

1. NAME: Must be creative, memorable, and professional. Examples:
   - For code review: "CodeCraft Pro", "PyReviewer Elite", "SyntaxMaster"
   - For content writing: "ContentForge", "WordSmith Pro", "NarrativeGenius"
   - For data analysis: "DataWiz", "InsightEngine", "AnalyticsPro"
   - NEVER use generic names like "Custom Agent", "AI Agent", or "New Agent"

2. ROLE: A professional job title (e.g., "Senior Python Developer", "Content Strategist")

3. DESCRIPTION: Clear 1-2 sentence explanation of what the agent does

4. PROMPT: Detailed instructions for the agent to follow when executing tasks

5. SETTINGS: Tune based on task type:
   - temperature: 0.1-0.3 for factual/precise tasks (code, math), 0.5-0.7 for balanced tasks, 0.8-1.0 for creative tasks
   - max_tokens: 1000-2000 for short outputs, 2000-4000 for detailed responses
   - top_p: 0.9 default, lower (0.5-0.7) for more focused outputs
   - max_iterations: 3-5 for simple tasks, 5-10 for complex multi-step tasks
"""

        llm = self._get_llm_client(model_id)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse response
        try:
            # Handle response content - may be string or have content attribute
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to extract JSON from response
            # Sometimes LLM wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            agent_spec = json.loads(content)
            return agent_spec
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return self._design_basic(user_prompt)

    def _design_basic(self, user_prompt: str) -> Dict[str, Any]:
        """Basic agent structure without LLM."""
        return {
            "name": "Custom Agent",
            "role": "Processing",
            "description": user_prompt[:200],
            "prompt": user_prompt,
            "input_schema": [],
            "output_schema": []
        }

    def update_from_prompt(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Update an existing agent based on user prompt while preserving identity.

        This method detects what fields the user wants to update and only
        modifies those fields, preserving the agent's name, ID, and other
        unchanged attributes.

        Args:
            existing_agent: The current agent definition dict
            user_prompt: User's natural language update request

        Returns:
            Updated agent definition dict with preserved identity
        """
        # Detect what the user wants to update
        update_intent = self._detect_update_fields(user_prompt)

        # Use LLM to generate updates for specified fields only
        try:
            updates = self._generate_field_updates(
                existing_agent, user_prompt, update_intent
            )
        except Exception as e:
            print(f"LLM update failed, applying basic updates: {e}")
            updates = self._apply_basic_updates(existing_agent, user_prompt, update_intent)

        # Merge updates into existing agent, preserving identity
        updated_agent = existing_agent.copy()

        # CRITICAL: Always preserve agent_id and name unless explicitly requested
        preserved_fields = ["agent_id", "name"]
        if "name" not in update_intent.get("fields_to_update", []):
            # Name should not change unless explicitly requested
            updates.pop("name", None)

        # Apply updates
        for key, value in updates.items():
            if key not in preserved_fields or key in update_intent.get("fields_to_update", []):
                updated_agent[key] = value

        # Update metadata
        if "metadata" not in updated_agent:
            updated_agent["metadata"] = {}
        updated_agent["metadata"]["updated_at"] = datetime.utcnow().isoformat()
        updated_agent["metadata"]["update_prompt"] = user_prompt

        return updated_agent

    def _detect_update_fields(self, user_prompt: str) -> Dict[str, Any]:
        """
        Detect which fields the user wants to update based on prompt keywords.

        Args:
            user_prompt: User's update request

        Returns:
            Dict with fields_to_update list and detected intent
        """
        prompt_lower = user_prompt.lower()

        fields_to_update = []

        # Tool-related keywords
        tool_keywords = ["tool", "tools", "add tool", "remove tool", "change tool",
                         "web search", "file reader", "code executor"]
        if any(kw in prompt_lower for kw in tool_keywords):
            fields_to_update.append("tools")

        # Description/purpose keywords
        desc_keywords = ["description", "purpose", "what it does", "goal", "objective"]
        if any(kw in prompt_lower for kw in desc_keywords):
            fields_to_update.append("description")

        # Role keywords
        role_keywords = ["role", "position", "job", "title"]
        if any(kw in prompt_lower for kw in role_keywords):
            fields_to_update.append("role")

        # Prompt/behavior keywords
        behavior_keywords = ["prompt", "behavior", "instruction", "system prompt",
                            "how it works", "act", "behave"]
        if any(kw in prompt_lower for kw in behavior_keywords):
            fields_to_update.append("prompt")

        # Settings keywords
        settings_keywords = ["temperature", "max_token", "setting", "configure",
                            "parameter", "iteration"]
        if any(kw in prompt_lower for kw in settings_keywords):
            fields_to_update.append("settings")

        # Name keywords (only if explicitly mentioned)
        name_keywords = ["rename", "change name", "new name", "call it", "named"]
        if any(kw in prompt_lower for kw in name_keywords):
            fields_to_update.append("name")

        # If no specific fields detected, assume description and prompt update
        if not fields_to_update:
            fields_to_update = ["description", "prompt"]

        return {
            "fields_to_update": fields_to_update,
            "original_prompt": user_prompt
        }

    def _generate_field_updates(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str,
        update_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate updates for specified fields.

        Args:
            existing_agent: Current agent definition
            user_prompt: User's update request
            update_intent: Dict with fields_to_update

        Returns:
            Dict with updated field values
        """
        fields_to_update = update_intent.get("fields_to_update", [])

        system_prompt = f"""You are an AI agent updater. The user wants to modify an existing agent.

EXISTING AGENT:
- Name: {existing_agent.get('name', 'Unknown')}
- Role: {existing_agent.get('role', 'Processing')}
- Description: {existing_agent.get('description', '')}
- Current Tools: {existing_agent.get('tools', [])}
- System Prompt: {existing_agent.get('prompt', '')[:500]}

FIELDS TO UPDATE: {fields_to_update}

Based on the user's request, provide updates ONLY for the specified fields.
Return a JSON object with ONLY the fields that need updating.

IMPORTANT RULES:
1. NEVER change the agent's name unless "name" is in the fields to update
2. Only return the fields listed in FIELDS TO UPDATE
3. Preserve the agent's core identity and purpose
4. For tools: return the complete new tools list (not just additions/removals)

Example response format:
{{"description": "Updated description", "tools": ["tool1", "tool2"]}}
"""

        llm = self._get_llm_client()
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object with only the updated fields."

        response = llm.invoke(full_prompt)

        # Parse response
        try:
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            updates = json.loads(content)

            # Validate that only requested fields are updated
            validated_updates = {}
            for field in fields_to_update:
                if field in updates:
                    validated_updates[field] = updates[field]

            return validated_updates

        except json.JSONDecodeError:
            return self._apply_basic_updates(existing_agent, user_prompt, update_intent)

    def _apply_basic_updates(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str,
        update_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply basic updates without LLM (fallback).

        Args:
            existing_agent: Current agent definition
            user_prompt: User's update request
            update_intent: Dict with fields_to_update

        Returns:
            Dict with basic updated values
        """
        fields_to_update = update_intent.get("fields_to_update", [])
        updates = {}

        if "description" in fields_to_update:
            # Append update context to description
            current_desc = existing_agent.get("description", "")
            updates["description"] = f"{current_desc} (Updated: {user_prompt[:100]})"

        if "prompt" in fields_to_update:
            # Append to system prompt
            current_prompt = existing_agent.get("prompt", "")
            updates["prompt"] = f"{current_prompt}\n\nAdditional instructions: {user_prompt}"

        return updates
