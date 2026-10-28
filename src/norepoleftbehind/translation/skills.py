"""Guide use of skill metadata and instructions already supplied in model context."""

DEFAULT_SKILL_PROMPT = (
    "Use a listed skill when its description is relevant to the request. "
    "Available skill names, descriptions, locations, and loaded instructions are supplied "
    "in the instructions or conversation context. Skills provide procedures and reference "
    "material; they are not callable functions. If a relevant skill's instructions are not "
    "already present, request an available skill-loading or file-reading tool using the "
    "provided name or location, then wait for its result before applying those instructions. "
    "Use only instructions actually supplied or loaded. Do not invent skills, locations, "
    "instructions, or extra tools. A skill name may appear in tool_calls only if that exact "
    "name is also a declared tool."
)
