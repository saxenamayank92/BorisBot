"""Deterministic LLM engine exception hierarchy."""


class LLMError(Exception):
    """Base error type for all engine/provider failures."""


class LLMTimeoutError(LLMError):
    """Provider call exceeded timeout limits."""


class LLMRetryableError(LLMError):
    """Transient provider failure that can be retried/failovered."""


class LLMCircuitOpenError(LLMError):
    """Provider circuit is open and request cannot be sent."""


class LLMAllProvidersFailed(LLMError):
    """No provider could successfully complete request."""


class LLMValidationError(LLMError):
    """Invalid request shape/inputs for LLM execution."""


class LLMStructuredError(LLMError):
    """LLM error carrying stable taxonomy class/code fields."""

    def __init__(self, message: str, *, error_class: str, error_code: str):
        super().__init__(message)
        self.error_class = error_class
        self.error_code = error_code


class LLMProviderUnhealthyError(LLMStructuredError):
    """Raised when provider state machine marks provider unhealthy."""

    def __init__(self, message: str = "llm provider unhealthy"):
        super().__init__(
            message,
            error_class="llm_provider",
            error_code="LLM_PROVIDER_UNHEALTHY",
        )


class LLMInvalidOutputError(LLMStructuredError):
    """Raised when planner response cannot be validated/repaired."""

    def __init__(self, message: str = "planner output invalid"):
        super().__init__(
            message,
            error_class="llm_invalid_output",
            error_code="LLM_OUTPUT_INVALID",
        )


class LLMTimeoutStructuredError(LLMStructuredError):
    """Raised for provider timeout with stable error taxonomy."""

    def __init__(self, message: str = "llm timeout"):
        super().__init__(
            message,
            error_class="llm_provider",
            error_code="LLM_TIMEOUT",
        )
