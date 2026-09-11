"""The persona the demo starts from.

`--text-prompt` shipped as "You are a helpful and friendly voice assistant." That string reads like
the obvious default and is the one thing a PersonaPlex fine-tune is least likely to have seen: an
audit of the earlier release's shards (374,140 records, 1,597 files) found no persona dropout at
all and no plain assistant anywhere in the role grid — "assistant", "helpful", "concierge" and
"receptionist" occur zero times in a role sentence. Off the distribution the model does not fall
back to neutral; it opens an arbitrary scene, which is what #18 records.

DEFAULT_PERSONA below is built to the shape that grid uses: a named role with a warm manner and a
subject it knows, closed by the generic framing sentence that ended 363,980 of those records. It is
the most domain-free corner of that shape rather than a generic assistant, because a generic
assistant is not something the fine-tune was shown.

  NOTE FOR THE TRAINING SIDE: this is carried over from the earlier release's audit. Confirm it
  against the current shards (r2b / spanturn v2) and replace it with a string sampled from those if
  the grid has moved; the point is that the default must exist in the data, not that it is this
  exact sentence.
"""

DEFAULT_PERSONA = (
    "You are Kelina, a kind and steady personal guide who helps people with everyday decisions "
    "and feelings. You can explain weighing pros and cons, choosing by your values, sleeping on a "
    "big choice. You are moshi, a capable voice assistant who also handles any everyday request "
    "— looking things up, booking, weather, reminders, directions — competently; the "
    "role above is your warm manner and the area you know best, not the only thing you help with."
)

# What the page puts in the profile fields when the user has not chosen. The router fills a missing
# location or timezone from the profile, so an empty one silently disables the time, weather and
# places tools; a default that matches the demo's own deployment is better than a blank box.
DEFAULT_LOCATION = "Seoul"
DEFAULT_TIMEZONE = "Asia/Seoul"
