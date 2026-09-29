# coding: utf-8
"""
Hospitality (hotel guest <-> agent SMS) flavor of VADER.

VADER's lexicon only scores single words, so a lot of what guests text a hotel
comes out as neutral: numeric stay ratings ("0. I never got a response"),
multi-word service failures ("tv remote was not working", "on hold for 15
minutes and gave up") and long, polite messages where one real problem is
buried under a lot of "thank you"s. This module layers three things on top of
the stock ``SentimentIntensityAnalyzer`` (which already loads the
hospitality-tuned ``vader_lexicon.txt``):

1. A 0-10 stay rating, when the guest gives one, is scored as a sentiment item.
2. Hotel complaint / praise phrases are scored as sentiment items, and
   negative items are weighted more heavily than positive ones, because a
   single unresolved problem matters more than a closing "thanks!".
3. When the text alone is a tough call (mixed or weak), an alert-worthiness
   prediction can be passed in to pull the score toward the side it implies.

``polarity_scores`` returns VADER's usual keys plus ``label`` so the numeric
``compound`` score and the label always agree.
"""
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# A complaint outweighs the same amount of praise in guest messaging.
NEGATIVE_WEIGHT = 1.8

# Valence for a stay rating on a 0-10 scale. 5 is omitted on purpose: guests
# use both 5-star and 10-point scales, so a bare "5" is ambiguous.
RATING_VALENCE = {0: -3.0, 1: -3.0, 2: -2.6, 3: -2.6, 4: -2.0, 6: -1.2, 7: -0.8, 8: -0.5}
ABOVE_SCALE_VALENCE = 1.5

ALERT_PRIOR = {
    "definitely alert-worthy": -1.0,
    "probably alert-worthy": -0.6,
    "not alert-worthy": 0.3,
}
ALERT_WEIGHT = 0.5
TOUGH_CALL_MARGIN = 0.5
MIXED_RATIO = 0.2

_SEGMENT_MARKER = re.compile(r"\(\s*\d+\s*/\s*\d+\s*\)")
# A leading "-" is usually a bullet ("-10, it was very nice"), not a negative rating.
_LEADING_RATING = re.compile(
    r"""^[\s"'“”‘’#*%&.,:;(-]*(?:re:\s*\)\s*)?[\s"'“”#*(-]*
        (?P<num>\d{1,2})(?:\.\d+)?(?![a-z\d])
        (?:\s*(?:/|out\s+of)\s*(?P<scale>5|10)\b)?
        (?!\s*(?:\d|:\d|am\b|pm\b|min|hour|hr|night|day|people|year|month|week|%|\$|th\b|st\b|nd\b|rd\b))""",
    re.IGNORECASE | re.VERBOSE)
_INLINE_RATING = re.compile(
    r"""\b(?:rate|rated|rating|give|giving|gave|say|score|grade)\b[^.!?\d]{0,25}?
        \b(?P<num>\d{1,2})(?:\.\d+)?\b
        (?:\s*(?:/|out\s+of)\s*(?P<scale>5|10)\b)?
        (?!\s*(?:am\b|pm\b|min|hour|hr|night|day|people|year|month|week|%|\$))""",
    re.IGNORECASE | re.VERBOSE)
# Guests often answer the survey with "0" when they never stayed; that zero is not a rating.
_NOT_A_RATING = re.compile(
    r"\bjust\s+kidding\b|\bj/?k\b|\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+not\s+(?:there|staying|at\s+(?:the\s+)?hotel|a\s+guest|checking\s+in)\b|"
    r"\b(?:didn'?t|did\s+not|never)\s+(?:stay|go|check\s+in|arrive)\b|\bhaven'?t\s+(?:arrived|checked\s+in|stayed)\b|"
    r"\bnot\s+(?:my\s+stay|coming\s+until|checking\s+in\s+until|at\s+(?:the\s+)?hotel)\b|\bwrong\s+(?:number|person)\b",
    re.IGNORECASE)

_NEG = r"(?:not|never|no|n't|nt|cannot)"
_BROKEN_THING = (r"(?:work|works|working|worked|function|functional|functioning|operable|turn\s+on|turn\s+off|"
                 r"flush|drain|cool|heat|connect|open|lock|close|stop)")

COMPLAINT_PHRASES = [
    (r"\b" + _NEG + r"\b(?:\W+\w+){0,3}?\W+" + _BROKEN_THING + r"\b", -1.8),
    (r"(?:n't|\bnot)\s+" + _BROKEN_THING + r"\b", -1.8),
    (r"\bnon-?working\b|\bout\s+of\s+order\b|\bstopped\s+working\b", -1.8),
    (r"\b(?:never|not|nobody|no\s+one|didn't|did\s+not|haven't|have\s+not|hasn't|has\s+not)\b(?:\W+\w+){0,3}?\W+"
     r"(?:respon\w*|repl\w*|answer\w*|showed|show\s+up|came|arrived|received|heard|followed\s+up|delivered|delived|"
     r"called\s+back|got\s+back)\b", -2.0),
    (r"\bno\s+(?:response|reply|replies|answer|call\s*back|follow[\s-]?up)\b", -2.0),
    (r"\b(?:did\s+not|didn't|didnt|never|couldn't|couldnt|could\s+not|can't|cant|cannot|unable\s+to)\s+"
     r"(?:get|receive|sleep|find|use|watch|access|finish|reach|take|order|change)\b", -1.4),
    (r"\bstill\s+(?:waiting|nothing|no\b|not\b|isn't|isnt|hasn't|hasnt|haven't|havent|broken|dirty|an?\s+issue)", -1.8),
    (r"\bon\s+hold\b|\bgave\s+up\b|\bgot\s+the\s+runaround\b|\bgo\s+figure\b", -1.6),
    (r"\bhad\s+to\s+(?:wait|call|ask|go\s+(?:down|back)|move|change|switch|walk|clean|sleep|pay|purchase|buy|find|"
     r"leave|cancel|shorten|request|jump|stand|put|wash|be\s+moved|check\s+out)\b", -1.3),
    (r"\bwait(?:ed|ing)?\b[^.!?]{0,30}?\b(?:\d+|two|three|four|five|several)\s*(?:\+\s*)?(?:min|mins|minutes|hour|hours|hrs)\b", -1.5),
    (r"\b\d+\s*(?:min|minute|hour)\s+wait\b|\bwait(?:ed)?\s+(?:too\s+)?long\b|\blong\s+(?:wait|lines?)\b", -1.5),
    (r"\b(?:ran|run|running)\s+out\b|\bnot\s+enough\b|\btoo\s+(?:many|few|long|loud|hot|cold)\b", -1.2),
    (r"\b(?:was|were|is|are|been|has|have|had)\s+(?:not|never)\s+(?:clean|cleaned|done|serviced|set\s*up|ready|"
     r"available|stocked|replaced|fixed|resolved|restocked|picked\s+up|made\s+up)\b", -1.6),
    (r"\b(?:isn't|isnt|wasn't|wasnt|aren't|weren't|hasn't|haven't)\s+(?:been\s+)?(?:clean|cleaned|done|serviced|"
     r"ready|available|fixed|resolved|right)\b", -1.6),
    (r"\bno\s+(?:hot\s+water|water|towels?|blankets?|pillows?|coffee|a/?c|a\.c|heat|internet|wifi|wi-fi|tv|remote|"
     r"room\s+service|service|room\s+available|cups|glasses|toilet\s+paper|soap|parking|screen|instructions|"
     r"information|seating|housekeeping|stopper)\b", -1.6),
    (r"\bkept\s+(?:us|me|my\s+\w+)\s+(?:up|awake)\b|\bup\s+(?:till|until)\s+\d|\bcouldn't\s+sleep\b", -1.8),
    (r"\b(?:checked|put)\s+(?:me\s+|us\s+)?into\s+an?\s+occupied\b|\broom\s+was\s+given\s+away\b", -2.0),
    (r"\b(?:switch|move|moved|change|changed)\s+(?:me\s+|us\s+)?(?:to\s+)?(?:another|a\s+different|new)\s+room\b|"
     r"\bchange\s+rooms\b|\bchanged\s+rooms\b", -1.2),
    (r"\b(?:was|were|got|been|still)\s+(?:charged|billed)\s+(?:for|me|us|twice|full|an?\s+(?:added|additional|extra))\b|"
     r"\bripped\s+off\b|\bprice\s+gouging\b", -1.6),
    (r"\b(?:contact(?:ing)?|call(?:ing)?|writ(?:e|ing)\s+(?:to|into)?)\s+(?:the\s+)?(?:corporate|company|headquarters)\b|"
     r"\b(?:bad|negative)\s+reviews?\b|\bwriting\s+a\s+review\b|\bfiled?\s+a\s+complaint\b", -1.8),
    (r"\bbed\s+bugs?\b|\bdust\s+mites?\b|\bfood\s+poisoning\b|\bfire\s+alarm\b", -2.2),
    (r"\bneeds?\s+(?:to\s+be\s+)?(?:service|fixed|repaired|cleaned|replaced|attention|work)\b|"
     r"\bwon't\s+stop\b|\bdoesn't\s+turn\s+off\b", -1.2),
    (r"\bwould\s+have\s+been\s+an?\s+\d+\s+if\b|\bif\s+it\s+was\s+not\s+for\b|\bwithholding\b", -1.5),
]

PRAISE_PHRASES = [
    (r"\bcouldn't\s+be\s+better\b|\bcould\s+not\s+be\s+better\b|\bcouldn't\s+have\s+been\s+(?:more|better)\b|"
     r"\bcould\s+not\s+have\s+been\s+(?:more|better)\b", 3.0),
    (r"\babove\s+(?:&|and)\s+beyond\b|\bcan't\s+say\s+enough\b|\bno\s+(?:complaints|issues|problems)\b", 2.0),
    (r"\ball\s+is\s+well\b|\bwill\s+(?:definitely\s+|surely\s+|certainly\s+)?(?:be\s+back|stay\s+again|return)\b|"
     r"\bstay\s+again\b|\bhighly\s+recommend\b", 1.5),
]

_COMPLAINT_REGEXES = [(re.compile(p, re.IGNORECASE), v) for p, v in COMPLAINT_PHRASES]
_PRAISE_REGEXES = [(re.compile(p, re.IGNORECASE), v) for p, v in PRAISE_PHRASES]


def extract_rating(text):
    """
    Return the guest's stay rating normalized to a 0-10 scale, or None.
    """
    stripped = _SEGMENT_MARKER.sub(" ", text).strip()
    if _NOT_A_RATING.search(stripped):
        return None
    match = _LEADING_RATING.match(stripped) or _INLINE_RATING.search(stripped)
    if not match:
        return None
    value = int(match.group("num"))
    if match.group("scale") == "5":
        value *= 2
    elif value > 12:
        return None
    return value


def rating_valence(rating):
    if rating is None:
        return 0.0
    if rating > 10:
        return ABOVE_SCALE_VALENCE
    return RATING_VALENCE.get(rating, 0.0)


def label_for(compound):
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


class HospitalitySentimentAnalyzer(SentimentIntensityAnalyzer):
    """
    Sentiment scoring for hotel guest SMS conversations.
    """

    def score_valence(self, sentiments, text):
        weighted = [s * NEGATIVE_WEIGHT if s < 0 else s for s in sentiments]
        for regex, valence in _COMPLAINT_REGEXES:
            weighted.extend([valence * NEGATIVE_WEIGHT] * len(regex.findall(text)))
        for regex, valence in _PRAISE_REGEXES:
            weighted.extend([valence] * len(regex.findall(text)))
        rating = extract_rating(text)
        if rating is not None:
            weighted.append(rating_valence(rating))
        return super(HospitalitySentimentAnalyzer, self).score_valence(weighted, text)

    def polarity_scores(self, text, alert_level=None):
        """
        Score ``text``. ``alert_level`` is an optional alert-worthiness
        prediction ("Definitely alert-worthy", "Probably alert-worthy",
        "Not alert-worthy"); it only moves the score when the text is a
        tough call, i.e. the message is mixed or its compound is weak.
        """
        scores = super(HospitalitySentimentAnalyzer, self).polarity_scores(text)
        compound = scores["compound"]
        prior = ALERT_PRIOR.get(str(alert_level).strip().lower()) if alert_level is not None else None
        if prior is not None:
            pos, neg = scores["pos"], scores["neg"]
            mixed = min(pos, neg) >= MIXED_RATIO * max(pos, neg) > 0
            # A "not alert-worthy" prediction says nothing about a message with no clear sentiment.
            weak = abs(compound) < TOUGH_CALL_MARGIN and prior < 0
            if mixed:
                # compound saturates near +/-1 on long messages, so weigh the pos/neg balance instead.
                compound = (1 - ALERT_WEIGHT) * (pos - neg) / (pos + neg) + ALERT_WEIGHT * prior
            elif weak:
                compound = (1 - ALERT_WEIGHT) * compound + ALERT_WEIGHT * prior
        scores["compound"] = round(compound, 4)
        scores["rating"] = extract_rating(text)
        scores["label"] = label_for(scores["compound"])
        return scores
