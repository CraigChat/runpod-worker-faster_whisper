# Most of this function derives from Kadda OK's TASMAS.
# https://github.com/KaddaOK/TASMAS/blob/main/assemble.py

import json
import os
import re
from operator import attrgetter
from typing import Dict, List, Optional, Tuple
import uuid
import warnings
from deepmultilingualpunctuation import PunctuationModel


DEFAULT_PUNC_CHUNK_SIZE = 230


def get_punc_chunk_size():
    env_value = os.getenv("PUNC_CHUNK_SIZE")
    if env_value is None:
        return DEFAULT_PUNC_CHUNK_SIZE
    try:
        return int(env_value)
    except ValueError:
        warnings.warn("Invalid PUNC_CHUNK_SIZE value; falling back to default.")
        return DEFAULT_PUNC_CHUNK_SIZE


def ends_with_break(text: str):
    last_character = text.strip()[-1]
    return last_character in ['.', '!', '?', '-', ',', '~']


class WordEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (Word, WordList)):
            return o.to_dict()
        return super().default(o)


class Word:
    def __init__(self, track, word, start, end):
        self.id = uuid.uuid4()
        self.track = track
        self.raw_word: str = word
        # self.word: str = word.strip()
        self.start = start
        self.end = end

    @property
    def ending_punc(self):
        match = re.search(r"(?<!\d)[.,;:!?](?!\d)$", self.raw_word.strip())
        return match.group(0) if match else "0"

    def replace_punc(self, new_punc):
        # Remove existing punctuation
        self.raw_word = re.sub(r"(?<!\d)[.,;:!?](?!\d)$", "", self.raw_word)
        normalized_punc = "" if new_punc is None else str(new_punc)
        if normalized_punc and normalized_punc != "0":
            self.raw_word += normalized_punc

    def to_dict(self):
        return {
            'track': self.track,
            'word': self.raw_word,
            'start': self.start,
            'end': self.end
        }


class WordList:
    def __init__(self, words: List[Word]):
        if len(set(word.track for word in words)) > 1:
            raise ValueError("All words in a WordList must belong to the same track")
        self.words = words

    @property
    def track(self):
        return self.words[0].track if self.words else None

    @property
    def start(self):
        return min(word.start for word in self.words)

    @property
    def end(self):
        return max(word.end for word in self.words)

    @property
    def text(self):
        return "".join(word.raw_word for word in self.words)
        # return " ".join(word.word for word in self.words)

    # @property
    # def pretty_text(self):
    #     return re.sub(r'(\w) ([-&]\w)', r'\1\2', self.text)

    def re_punctuate(self, model: PunctuationModel):
        # word with last index that can be punctuated
        processable_words: List[Tuple[str, int]] = []

        for i in range(len(self.words)):
            word = self.words[i]
            clean_word = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "",  word.raw_word.strip()) 
            space_behind = word.raw_word.startswith(" ")
            if not space_behind and len(processable_words) > 0:
                # append to last processed word
                last_word = processable_words[len(processable_words) - 1]
                processable_words[len(processable_words) - 1] = (last_word[0] + clean_word, i)
            else:
                processable_words.append((clean_word, i))

        chunk_size = get_punc_chunk_size()
        results = model.predict([w[0] for w in processable_words], chunk_size)

        change_count = 0
        for i in range(len(results)):
            result = results[i]
            punc = result[1]
            word_index = processable_words[i][1]
            word = self.words[word_index]
            old_punc = word.ending_punc
            new_punc = "0" if punc in (None, "", "0") else str(punc)
            if old_punc != new_punc:
                change_count += 1
                word.replace_punc(new_punc)

        return change_count

    def to_dict(self):
        return {
            'track': self.track,
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'words': [word.to_dict() for word in self.words]
        }


def normalize_and_merge_words(words: List[Word]):
    # First, normalize words by grouping them in word lists according to the track it belongs to
    normal_words: List[WordList] = []
    current_sentences: Dict[int, WordList] = {}

    for word in words:
        if word.track not in current_sentences or current_sentences[word.track] is None:
            current_sentences[word.track] = WordList([word])
        else:
            current_sentences[word.track].words.append(word)

        last_character = word.raw_word.strip()[-1]
        if last_character in ['.', '!', '?', '-', ',', '~'] and current_sentences[word.track]:
            normal_words.append(current_sentences[word.track])
            current_sentences.pop(word.track)

    # catch any leftovers
    leftovers = [v for k, v in current_sentences.items() if v is not None]
    leftovers.sort(key=lambda x: x.start)
    normal_words.extend(leftovers)

    # Now, merge adjacent same-track word lists together
    current_chunk: Optional[WordList] = None
    merged_word_lists: List[WordList] = []

    for next_list in normal_words:
        if current_chunk is None:
            current_chunk = next_list
            continue
        sentence_gap = next_list.start - current_chunk.end
        if next_list.track != current_chunk.track or sentence_gap > 2:
            merged_word_lists.append(current_chunk)
            current_chunk = next_list
        else:
            current_chunk.words.extend(next_list.words)
    if current_chunk is not None:
        merged_word_lists.append(current_chunk)

    return merged_word_lists


def get_desynced_words(word_lists: List[WordList]):
    running_time = 0
    desynced_lists: List[Tuple[WordList, float]] = []

    for word_list in word_lists:
        if word_list.start < running_time - 5:
            desynced_lists.append((word_list, round(running_time - word_list.start, 2)))
        running_time = word_list.start

    return desynced_lists


# Add ellipsis, will be inserted after 5 seconds of silence
def add_ellipsis_to_words(track_words: List[Word]):
  for i in range(1, len(track_words)):
      word = track_words[i]
      last_word = track_words[i - 1]
      if word.start - last_word.end > 5 and not ends_with_break(last_word.raw_word):
          last_word.raw_word = last_word.raw_word + "..."

def correct_words(all_words: List[Word], model: PunctuationModel, logger):
    sorted_words = sorted(all_words, key=attrgetter('start', 'track', 'end'))
    normalized_lines = normalize_and_merge_words(sorted_words)
    desynced_lists = get_desynced_words(normalized_lines)
    change_count = 0
    if len(desynced_lists) > 0:
        logger.info(f"Found {len(desynced_lists)} lines to be out of sync")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            words_to_check: List[Word] = []
            for item, _ in desynced_lists:
                changes = item.re_punctuate(model)
                if changes > 0:
                    words_to_check.extend(item.words)
                    change_count += 1
            logger.info(f"Changed {change_count} lines")
            if change_count > 0:
                word_by_id: Dict[uuid.UUID, Word] = {}
                for word in words_to_check:
                    word_by_id[word.id] = word
                for og_word in all_words:
                    if og_word.id in word_by_id:
                        og_word.raw_word = word_by_id[og_word.id].raw_word

                sorted_words = sorted(all_words, key=attrgetter('start', 'track', 'end'))
                normalized_lines = normalize_and_merge_words(sorted_words)
    
    # logger.info(f"Repunctuating {len(normalized_lines)} lines")
    # for line in normalized_lines:
    #     repunctuated = model.restore_punctuation(line.text)
    #     replace_corrected_words(line, repunctuated, all_words)

    return ([wl.to_dict() for wl in normalized_lines], change_count)