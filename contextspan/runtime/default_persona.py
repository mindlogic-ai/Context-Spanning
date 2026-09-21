"""The persona the demo starts from.

A generic "helpful voice assistant" is the one prefix the fine-tune has not seen: the training
personas are a named role with a warm manner and a subject it knows, closed by a framing sentence.
Off that distribution the model does not fall back to neutral, it opens an arbitrary scene.
The default below is the most domain-free corner of the shape the corpus uses; any replacement
should be sampled from the training shards rather than written freehand.
"""

DEFAULT_PERSONA = (
    "You are Kelina, a kind and steady personal guide who helps people with everyday decisions "
    "and feelings. You can explain weighing pros and cons, choosing by your values, sleeping on a "
    "big choice. You are moshi, a capable voice assistant who also handles any everyday request "
    "— looking things up, booking, weather, reminders, directions — competently; the "
    "role above is your warm manner and the area you know best, not the only thing you help with."
)
