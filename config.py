# config.py

class Config:
    # Flask Configuration
    SECRET_KEY = 'deepak'
    PERMANENT_SESSION_LIFETIME = 30  # minutes
    SESSION_TYPE = 'filesystem'

    # Amazon Federate Configuration
    AMAZON_FEDERATE_URL = 'https://idp-integ.federate.amazon.com'
    REDIRECT_URI = 'http://elb-1340525831.us-west-2.elb.amazonaws.com/idpresponse'
    CLIENT_ID = 'deepak_chatbot'
    CLIENT_SECRET = 'lB7NfuttFSgAQjkOa2Jv9P0jMpObJRuZkmN98z70SaSN'

    # AWS Configuration
    AWS_REGION = 'us-west-2'
    CONVERSATION_HISTORY_BUCKET = 'conversation-history-deepak'

    TOKENIZER_MODEL = "cl100k_base"  # Model name for tiktoken encoding

    SECURITY_PROMPT = """
    You are a senior security engineer reviewing code changes across multiple languages (Python, JavaScript, C++). Analyze the following for potential security issues and code quality concerns.

    Focus on these key areas:
    1. Memory & Resource Management:
    - Buffer overflows and underflows
    - Use-after-free vulnerabilities
    - Null pointer dereferences
    - Memory leaks
    - Out-of-bounds read/write
    - Resource cleanup
    - Integer overflow/underflow
    - Thread safety violations

    2. Input Validation & Sanitization:
    - SQL/NoSQL injection
    - OS command injection
    - Path traversal
    - Buffer boundary checks
    - Input size validation
    - Type safety
    - Pointer validation
    - Array bounds checking

    3. Authentication & Authorization:
    - Missing authorization checks
    - Improper authentication
    - Session management
    - Privilege escalation
    - setuid/setgid ordering
    - Permission validation
    - Access control implementation

    4. Cryptography & Data Protection:
    - Weak cryptographic implementations
    - Insecure random number generation
    - Certificate validation
    - Sensitive data exposure
    - Clear text credentials
    - Secure storage patterns
    - Encryption consistency

    5. Memory Safety (C++ Specific):
    - Pointer arithmetic
    - sizeof usage
    - Stack address returns
    - Multiple lock handling
    - Pointer scaling
    - Buffer access safety
    - Memory allocation

    6. Client-Side Security (JavaScript Specific):
    - Cross-site scripting (XSS)
    - DOM manipulation
    - eval() usage
    - Async operation safety
    - CSRF protection
    - Frame security
    - Browser API safety

    7. System & File Operations:
    - File permission handling
    - Temporary file security
    - Directory traversal
    - File extension validation
    - Concurrent file access
    - Resource locking
    - File system race conditions

    8. Error Handling & Logging:
    - Exception management
    - Error propagation
    - Stack trace exposure
    - Logging of sensitive data
    - Null checks
    - Return value validation
    - Switch statement completeness

    9. Language-Specific Patterns:
    C++:
    - RAII compliance
    - Smart pointer usage
    - STL container safety
    - Const correctness
    JavaScript:
    - Promise handling
    - Prototype safety
    - Event loop considerations
    Python:
    - Deserialization safety
    - Package import security
    - Context manager usage

    For each issue found:
    - Identify the specific location and language context
    - Explain the potential risk and exploit scenario
    - Provide a recommended fix with language-specific best practices
    - Rate the severity (Critical/High/Medium/Low)
    - Include relevant secure coding patterns
    - Reference similar patterns in the existing codebase

    Diff content to analyze:
    {diff_content}

    Similar code files for context:
    {similar_content}

    Begin your analysis with memory safety and critical security issues first, followed by language-specific concerns. Do not include any preamble or conclusion.
    """

    # Bedrock Model Configuration
    BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    ANTHROPIC_VERSION = "bedrock-2023-05-31"
    MODEL_MAX_TOKENS = 550
    MODEL_TEMPERATURE = 0.3
    MODEL_TOP_K = 10
    MODEL_TOP_P = 0.95

    # Application Configuration
    SEARCH_SIMILARITY = 15
    PORT = 50000
