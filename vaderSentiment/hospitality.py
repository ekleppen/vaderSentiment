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
NEGATIVE_WEIGHT = 2.0

# Valence for a stay rating on a 0-10 scale. 5 is omitted on purpose: guests
# use both 5-star and 10-point scales, so a bare "5" is ambiguous.
RATING_VALENCE = {0: -3.0, 1: -3.0, 2: -2.8, 3: -2.8, 4: -2.4, 6: -1.8, 7: -1.4, 8: -0.9, 9: -0.3}
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
_CASUAL_RATING = re.compile(
    r"""\b(?:so\s+far|about|solid|probably|maybe|overall|stay\s+(?:is|was|has\s+been)|it'?s\s+been|it\s+is|it'?s|
        we're\s+at|i'?d\s+say|call\s+it)\s+(?:an?\s+)?(?P<num>\d{1,2})(?:\.\d+)?\b
        (?:\s*(?:/|out\s+of)\s*(?P<scale>5|10)\b)?
        (?!\s*(?:\d|:\d|am\b|pm\b|min|hour|hr|night|day|people|year|month|week|degree|dollar|bucks|%|\$|th\b|st\b|nd\b|rd\b))""",
    re.IGNORECASE | re.VERBOSE)
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10}
_LEADING_WORD_RATING = re.compile(r"^[\s\"'“”#*(]*(?P<word>" + "|".join(_NUMBER_WORDS) + r")\b[\s.!,:;)-]",
                                  re.IGNORECASE)
_CONTRAST_WORDS = {"but", "however"}
# "No, I'm good" answers the agent's question; VADER would otherwise read it as negating "good".
_ANSWER_NO = re.compile(
    r"^(\W*)no\b[\s,.!]*(?=(?:i'?m|i\s+am|we'?re|we\s+are|it'?s|all|everything|thanks?|thank|we|i)\b)",
    re.IGNORECASE)
# Guests often answer the survey with "0" when they never stayed; that zero is not a rating.
_NOT_A_RATING = re.compile(
    r"\bjust\s+kidding\b|\bj/?k\b|\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+not\s+(?:there|staying|at\s+(?:the\s+)?hotel|a\s+guest|checking\s+in)\b|"
    r"\b(?:didn'?t|did\s+not|never)\s+(?:stay|go|check\s+in|arrive)\b|\bhaven'?t\s+(?:arrived|checked\s+in|stayed)\b|"
    r"\bnot\s+(?:my\s+stay|coming\s+until|checking\s+in\s+until|at\s+(?:the\s+)?hotel)\b|\bwrong\s+(?:number|person)\b",
    re.IGNORECASE)

_NEG = r"(?:\b(?:not|never|no|nt|cannot)|n't)"
_BROKEN_THING = (r"(?:work|works|working|worked|function|functional|functioning|operable|turn\s+on|turn\s+off|"
                 r"flush|drain|draining|cool|cooling|heat|heating|connect|open|lock|close|stop|charge|charging|"
                 r"stay\s+(?:closed|down|on|open)|come\s+on|produce|producing)")
_FAILED_TO = (r"(?:delivered|acted\s+on|addressed|advised|informed|cleaned|fixed|replaced|repaired|returned|honored|"
              r"refilled|restocked|emptied|removed|picked\s+up|made\s+up|serviced|resolved|sorted|done|brought|met)")

COMPLAINT_PHRASES = [
    (_NEG + r"\b(?:\W+\w+){0,3}?\W+" + _BROKEN_THING + r"\b", -1.8),
    (r"(?:n't|\bnot)\s+" + _BROKEN_THING + r"\b", -1.8),
    (r"\bnon-?working\b|\bout\s+of\s+order\b|\bstopped\s+working\b", -1.8),
    (r"\b(?:never|not|nobody|no\s+one|didn't|did\s+not|haven't|have\s+not|hasn't|has\s+not)\b(?:\W+\w+){0,3}?\W+"
     r"(?:respon\w*|repl\w*|answer\w*|showed|show\s+up|came|come|took|arrived|received|heard|followed\s+up|delivered|"
     r"delived|called\s+back|got\s+back)\b", -2.0),
    (r"\bmessed\s+up\b|\bmess\s+up\b|\bscrewed\s+up\b|\bdropped\s+the\s+ball\b|\bnot\s+up\s+to\s+(?:par|standard)\b", -1.8),
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
    (r"(?<!need\s)\bno\s+(?:hot\s+water|water|towels?|blankets?|pillows?|coffee|a/?c|a\.c|heat|internet|wifi|wi-fi|tv|remote|"
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
    # Concessions: guests soften a complaint with praise, but the complaint is the point of the message.
    (r"\b(?:only|main|biggest)\s+(?:minor\s+|small\s+|real\s+)?(?:reason|thing|issue|problem|prob|complaint|concern|"
     r"item|downside|negative|con|drawback|area\s+of\s+concern|suggestion)\b|\bonly\s+because\b|"
     r"\bone\s+(?:minor\s+|small\s+)?(?:issue|problem|complaint|concern|area\s+of\s+concern|downside|drawback)\b|"
     r"\breason\s+for\s+the\s+\d", -1.5),
    (r"\b(?:except|besides|beside|apart\s+from|aside\s+from|other\s+than)\b|\botherwise\b", -1.0),
    (r"\b(?:almost|nearly)\s+an?\s+(?:10|ten)\b|\bwould\s+(?:of|have)\s+been\s+an?\s+(?:10|ten)\b|"
     r"\b(?:taking|take|took|minus)\s+(?:off\s+)?(?:\d|one|two|a)\s+points?\b|\b-\s?\d\s+points?\b", -1.5),
    (r"\bwould\s+(?:have\s+been|be)\s+(?:nice|better|great|helpful)\b|\bcould\s+(?:be|have\s+been|use)\s+(?:better|more|some)\b|"
     r"\bwish\s+(?:the|it|they|there|you|we|i\s+had|i\s+could\s+say)\b|\bmight\s+want\s+to\b|\byou\s+may\s+want\s+to\b|\bplease\s+fix\b", -1.0),
    # Things that were promised or requested and did not happen.
    (r"\b(?:wasn't|wasnt|weren't|werent|was\s+not|were\s+not|isn't|isnt|hasn't|hasnt|haven't|havent|has\s+not|"
     r"have\s+not|never|not\s+been|never\s+been)\s+(?:been\s+|being\s+|properly\s+|yet\s+)?" + _FAILED_TO + r"\b", -1.6),
    (r"\b(?:that|it)\s+(?:didn't|did\s+not|never)\s+happen\b|\bbut\s+(?:it|they|he|she)\s+(?:wasn't|weren't|didn't|never)\b|"
     r"\bsupposed?\s+to\b[^.!?]{0,40}\b(?:and|but)\s+(?:did\s+not|didn't|never|wasn't)\b", -1.6),
    (r"\b(?:asked|called|requested|reported|told|texted|messaged)\b[^.!?]{0,40}?\b(?:twice|again|two\s+times|"
     r"three\s+times|several\s+times|multiple\s+times|(?:\d|two|three)\s+(?:times|days))\b|"
     r"\bbeen\s+(?:\d+|two|three|several)\s+days\s+since\b", -1.5),
    (r"\b(?:took|takes|taking)\s+(?:over\s+|almost\s+|about\s+|more\s+than\s+|nearly\s+)?(?:\d+|an?|two|three|several)\s*"
     r"(?:\+\s*)?(?:min|mins|minutes|hour|hours|hrs|days|attempts|tries)\b|\b(?:took|takes)\s+forever\b|"
     r"\b(?:several|multiple|many)\s+attempts\b", -1.4),
    (r"\b(?:burnt|burned|blown)\s+out\b|\bcut(?:ting|s)?\s+(?:in\s+and\s+)?out\b|\bno\s+signal\b|\bacting\s+up\b|"
     r"\b(?:internet|wifi|wi-fi|power|tv|cable)\s+(?:was|is|went|keeps\s+going)\s+(?:out|down)\b", -1.4),
    (r"\b(?:throwing|threw|thrown)\s+up\b|\bhair\s+(?:all\s+over|in\s+the|on\s+the|everywhere)\b|\bfruit\s+flies\b", -2.0),
    (r"\b(?:don't|doesn't|didn't|dont|doesnt|didnt)\s+(?:have|offer)\b|"
     r"\bthere\s+(?:wasn't|weren't|isn't|aren't|was\s+no|were\s+no|is\s+no|are\s+no)\b", -0.8),
    (r"\bno\s+(?:\w+\s+)?(?:microwave|refrigerator|fridge|fan|bath|bathtub|tub|hair\s*dryer|iron|robes?|hangers|ice|"
     r"lights?|power|skillet|pans?|utensils|dishes|shampoo|conditioner|lotion|kleenex|tissues|toiletries|amenities)\b", -1.2),
    (r"\b(?:a\s+(?:little|bit|tad)|kind\s+of|kinda|somewhat|pretty|too|very|really|so|way\s+too)\s+(?:too\s+)?"
     r"(?:warm|hot|cold|slow|small|old|loud|noisy|pricey|expensive|dated|smoky|crowded|soft|hard|firm|rough|dirty|"
     r"dusty|thin|stuffy|humid|chilly|dark|busy)\b", -1.0),
    (r"\b(?:not|isn't|wasn't|aren't|n't)\s+(?:\w+\s+)?(?:hot|cold|clean|warm|cool|bright|quiet|comfortable)\s+enough\b|"
     r"\b(?:isn't|wasn't|not)\s+(?:producing|making|getting)\s+enough\b", -1.2),
    (r"\bcan't\s+(?:seem\s+to|figure\s+out)\b|\bcannot\s+(?:seem\s+to|figure\s+out)\b|\bcould\s+not\s+figure\s+out\b|"
     r"\bcouldn't\s+figure\s+out\b", -1.2),
]

PRAISE_PHRASES = [
    (r"\bcouldn't\s+be\s+better\b|\bcould\s+not\s+be\s+better\b|\bcouldn't\s+have\s+been\s+(?:more|better)\b|"
     r"\bcould\s+not\s+have\s+been\s+(?:more|better)\b", 3.0),
    (r"\babove\s+(?:&|and)\s+beyond\b|\bcan't\s+say\s+enough\b|\bno\s+(?:complaints|issues|problems)\b", 2.0),
    (r"\b(?:zero|not\s+a\s+single|not\s+one)\s+(?:complaints?|issues?|problems?)\b|"
     r"\b(?:can't|cant|cannot|can\s+not)\s+(?:even\s+)?thank\s+(?:\w+\s+){0,3}?enough\b", 2.0),
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
    match = _LEADING_RATING.match(stripped) or _INLINE_RATING.search(stripped) or _CASUAL_RATING.search(stripped)
    if not match:
        word = _LEADING_WORD_RATING.match(stripped)
        return _NUMBER_WORDS[word.group("word").lower()] if word else None
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

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # Like VADER's "but" rule, extended to "however": what follows the contrast outweighs what precedes it.
        words = [str(w).lower() for w in words_and_emoticons]
        pivot = next((i for i, w in enumerate(words) if w in _CONTRAST_WORDS), None)
        if pivot is None:
            return sentiments
        return [s * 0.5 if i < pivot else s * 1.5 if i > pivot else s for i, s in enumerate(sentiments)]

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
        scores = super(HospitalitySentimentAnalyzer, self).polarity_scores(_ANSWER_NO.sub(r"\1", str(text)))
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
