from enum import Enum

class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"

print(UsageContext.UNKNOWN_CONTEXT)
print(UsageContext.LLM_CLASS)
print(UsageContext.API_SERVER)
print(UsageContext.OPENAI_API_SERVER)
print(UsageContext.OPENAI_BATCH_RUNNER)
print(UsageContext.ENGINE_CONTEXT)