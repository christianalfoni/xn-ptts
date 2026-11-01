//! Word timings from the flow LM's attention over the text it was prompted with.
//!
//! A frame is one step of the flow LM, `1 / frame_rate` seconds of audio (80 ms at the
//! 12.5 Hz Phonon codec). While generating, the new frame's query attends over the text
//! tokens sitting in the KV cache. [`crate::transformer::TextSpan`] reads that row, averaged
//! over the heads of one layer. [`WordClock`] walks the rows monotonically — at most one
//! text token forward per frame, and only when the next token's weight exceeds the current
//! one's — and turns the path into [`WordTiming`]s for the words of the caller's text.
//!
//! The walk is the one the browser highlighter this was built from used, and it is
//! prefix-stable: a word's time does not move once it has been emitted, so timings can be
//! reported as frames arrive. Nothing here runs unless a caller asks. The attention read
//! itself lives in the transformer and is skipped entirely when no span is passed.
//!
//! Normalization and chunking rewrite the text the model sees. [`map_utterance`] lines the
//! model's tokens back up with the words of the text the caller passed, before that
//! rewriting, so a timing's offsets address that text.

use crate::Tokenizer;
use crate::plan::Chunk;
use crate::tts_model::prepare_text_prompt;
use crate::{Error, Result};

/// One word of the caller's text, and when it is spoken.
///
/// `start` and `end` are Unicode scalar offsets into the text that was synthesized, before
/// normalization: `text.chars().skip(start).take(end - start)` is `word`. Times are seconds
/// from the start of the utterance's audio, on the model's frame grid.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct WordTiming {
    pub word: String,
    pub start: usize,
    pub end: usize,
    /// Seconds from the start of the utterance.
    pub from: f32,
    /// Seconds from the start of the utterance. Equal to `from` for a word the alignment
    /// never landed on, so it takes no time of its own.
    pub to: f32,
}

/// A word of the caller's text, with both offset kinds the mapping needs.
#[derive(Clone, Debug)]
struct OrigWord {
    word: String,
    start_byte: usize,
    end_byte: usize,
    start: usize,
    end: usize,
}

/// Which original word each text token of each chunk attends for.
pub struct UtteranceMap {
    words: Vec<OrigWord>,
    /// Parallel to the chunks [`map_utterance`] was given: one entry per text token, the
    /// original word it belongs to, or `None` for a token that is not a word (a period the
    /// prompt preparation added, for example).
    token_words: Vec<Vec<Option<u32>>>,
}

impl UtteranceMap {
    /// A clock that turns attention rows into [`WordTiming`]s, in utterance order.
    ///
    /// `frame_rate` is the codec's, `TTSConfig::mimi.frame_rate`: a frame lasts
    /// `1 / frame_rate` seconds.
    pub fn clock(&self, frame_rate: f64) -> WordClock {
        WordClock {
            words: self.words.clone(),
            token_words: self.token_words.clone(),
            chunk: 0,
            token_word: Vec::new(),
            frame: 0,
            frame_rate,
            prev_token: 0,
            open: None,
            cursor: 0,
        }
    }

    /// The original words, in order.
    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(|word| word.word.as_str())
    }
}

/// Walks attention rows into word timings as frames arrive.
///
/// Feed each chunk with [`Self::begin_chunk`] before its rows, [`Self::push`] once per frame,
/// and [`Self::end_chunk`] when the chunk's audio is finished. [`Self::push`] returns the
/// words whose end time that frame fixed; [`Self::end_chunk`] returns the words still open,
/// including the last one, which runs to the end of the chunk.
pub struct WordClock {
    words: Vec<OrigWord>,
    token_words: Vec<Vec<Option<u32>>>,
    chunk: usize,
    token_word: Vec<Option<u32>>,
    frame: usize,
    frame_rate: f64,
    prev_token: usize,
    open: Option<OpenWord>,
    /// Next original-word index to emit. Gaps the path skipped are emitted with no duration
    /// of their own, so every word of the text is reported and the list stays in order.
    cursor: usize,
}

struct OpenWord {
    index: usize,
    from_frame: usize,
}

impl WordClock {
    /// The next chunk's token-to-word map. Past the last chunk it records nothing.
    pub fn begin_chunk(&mut self) {
        self.token_word = self.token_words.get(self.chunk).cloned().unwrap_or_default();
        self.chunk += 1;
        // Each chunk's tokens start again at 0. The frame clock does not.
        self.prev_token = 0;
    }

    /// One frame's attention row over the chunk's text tokens, in token order.
    pub fn push(&mut self, row: &[f32]) -> Vec<WordTiming> {
        let token = step_token(self.prev_token, row);
        self.prev_token = token;
        let word = self.token_word.get(token).copied().flatten().map(|index| index as usize);
        let frame = self.frame;
        self.frame += 1;
        match word {
            Some(index) => self.advance_to(index, frame),
            // A token with no word of its own, such as the period a prompt gains, keeps the
            // word already open so the tail of the chunk still belongs to it.
            None => Vec::new(),
        }
    }

    /// Close the chunk: the open word runs to the end of its audio, and any word of this
    /// chunk the path never reached is reported at that boundary.
    pub fn end_chunk(&mut self) -> Vec<WordTiming> {
        let last = self.token_word.iter().flatten().copied().max().map(|index| index as usize);
        let mut out = Vec::new();
        if let Some(open) = self.open.take() {
            out.extend(self.emit_gap(open.index, open.from_frame));
            out.push(self.timing(open.index, open.from_frame, self.frame));
            self.cursor = open.index + 1;
        }
        if let Some(last) = last {
            out.extend(self.emit_gap(last + 1, self.frame));
        }
        out
    }

    fn advance_to(&mut self, index: usize, frame: usize) -> Vec<WordTiming> {
        if let Some(open) = &self.open
            && open.index == index
        {
            return Vec::new();
        }
        // A backwards step would be the path leaving a word and coming back. The token walk
        // does not do that; ignore it if a map ever did.
        if let Some(open) = &self.open
            && index < open.index
        {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(open) = self.open.take() {
            out.extend(self.emit_gap(open.index, open.from_frame));
            out.push(self.timing(open.index, open.from_frame, frame));
            self.cursor = open.index + 1;
        }
        out.extend(self.emit_gap(index, frame));
        self.open = Some(OpenWord { index, from_frame: frame });
        out
    }

    fn emit_gap(&mut self, until: usize, frame: usize) -> Vec<WordTiming> {
        let mut out = Vec::new();
        while self.cursor < until && self.cursor < self.words.len() {
            out.push(self.timing(self.cursor, frame, frame));
            self.cursor += 1;
        }
        out
    }

    fn timing(&self, index: usize, from_frame: usize, to_frame: usize) -> WordTiming {
        let word = &self.words[index];
        WordTiming {
            word: word.word.clone(),
            start: word.start,
            end: word.end,
            from: secs(from_frame, self.frame_rate),
            to: secs(to_frame, self.frame_rate),
        }
    }
}

/// The token a frame attends, given the token the previous frame attended.
///
/// Stays put, or moves one token forward when that token's weight is the larger of the two.
/// An empty row leaves `prev` unchanged.
pub fn step_token(prev: usize, row: &[f32]) -> usize {
    if row.is_empty() {
        return prev;
    }
    let prev = prev.min(row.len() - 1);
    let next = (prev + 1).min(row.len() - 1);
    if next != prev && row[next] > row[prev] { next } else { prev }
}

fn secs(frame: usize, frame_rate: f64) -> f32 {
    if frame_rate <= 0.0 {
        return 0.0;
    }
    let millis = (frame as f64 / frame_rate) * 1_000.0;
    (millis.round() as f32) / 1_000.0
}

/// Map `chunks` of `text` back onto the words of `text`.
///
/// `text` is what the caller passed, before normalization and before the chunks were cut.
/// `chunks` are what [`crate::plan::chunks`] produced from it, in order, so each chunk's text
/// is already normalized. The tokenizer is the one that encoded them.
pub fn map_utterance(
    text: &str,
    chunks: &[Chunk],
    tokenizer: &dyn Tokenizer,
) -> Result<UtteranceMap> {
    let words = words_in(text);
    let mut cursor = 0usize;
    let orig_content = content_of(text, &words);
    let mut token_words = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let spoken_words = whitespace_words(&chunk.text);
        let spoken_content = content_of(&chunk.text, &spoken_spans(&chunk.text));
        let (origins, advance) =
            align_content(&orig_content[cursor..], &spoken_content, spoken_words.len());
        cursor = (cursor + advance).min(orig_content.len());
        let prepared = prepare_text_prompt(&chunk.text).0;
        let spans = token_spans(tokenizer, &chunk.tokens, &prepared)?;
        let prepared_at = prepared_to_chunk(&chunk.text, &prepared);
        let mut tokens = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            // A piece often begins on the space before its word (`▁the`). The first byte then
            // sits between words; walk the span until a byte lands inside one.
            let origin = (end > start)
                .then(|| {
                    (start..end).find_map(|byte| {
                        let chunk_byte = prepared_at.get(byte).copied().flatten()?;
                        let index = word_index(&spoken_words, chunk_byte)?;
                        origins.get(index).copied().flatten()
                    })
                })
                .flatten();
            tokens.push(origin);
        }
        token_words.push(tokens);
    }
    Ok(UtteranceMap { words, token_words })
}

fn token_spans(
    tokenizer: &dyn Tokenizer,
    tokens: &[u32],
    prepared: &str,
) -> Result<Vec<(usize, usize)>> {
    let spans = tokenizer.encode_spans(prepared).map_err(Error::from)?;
    let known = spans.len() == tokens.len()
        && spans.iter().zip(tokens).all(|(span, id)| span.0 == *id)
        && spans.iter().any(|span| span.2 > span.1);
    if known {
        return Ok(spans.into_iter().map(|(_, start, end)| (start, end)).collect());
    }
    Ok(even_spans(tokens.len(), prepared))
}

/// A content character of `text`, and which of `words` (byte ranges) it falls in.
struct Content {
    ch: char,
    word: u32,
}

fn content_of(text: &str, words: &[OrigWord]) -> Vec<Content> {
    let mut out = Vec::new();
    for word in words {
        for ch in text[word.start_byte..word.end_byte].chars() {
            if let Some(ch) = fold_char(ch) {
                out.push(Content { ch, word: word_index_of(words, word) });
            }
        }
    }
    out
}

fn word_index_of(words: &[OrigWord], word: &OrigWord) -> u32 {
    words.iter().position(|other| other.start_byte == word.start_byte).unwrap_or(0) as u32
}

/// Align `spoken` content onto a window of `orig`, starting at the beginning of `orig`.
///
/// Returns the original word each spoken word landed on, and how many `orig` content
/// characters the match consumed.
fn align_content(
    orig: &[Content],
    spoken: &[Content],
    n_spoken_words: usize,
) -> (Vec<Option<u32>>, usize) {
    let window = spoken.len().saturating_mul(4).saturating_add(48).max(spoken.len());
    let orig = &orig[..window.min(orig.len())];
    let (n, m) = (orig.len(), spoken.len());
    if m == 0 || n_spoken_words == 0 {
        return (vec![None; n_spoken_words], 0);
    }
    // dp[i][j]: cost of consuming orig[..i] and spoken[..j]. Trailing orig is free, so the
    // match can stop at the first sentence instead of dragging in the rest of the text.
    let cols = m + 1;
    let mut dp = vec![0u32; (n + 1) * cols];
    let mut from = vec![0u8; (n + 1) * cols];
    let at = |i: usize, j: usize| i * cols + j;
    for i in 1..=n {
        dp[at(i, 0)] = i as u32;
        from[at(i, 0)] = 1; // delete orig
    }
    for j in 1..=m {
        dp[at(0, j)] = j as u32;
        from[at(0, j)] = 2; // insert spoken
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = dp[at(i - 1, j - 1)] + u32::from(orig[i - 1].ch != spoken[j - 1].ch);
            let del = dp[at(i - 1, j)] + 1;
            let ins = dp[at(i, j - 1)] + 1;
            let (cost, how) = if sub <= del && sub <= ins {
                (sub, 0u8)
            } else if del <= ins {
                (del, 1)
            } else {
                (ins, 2)
            };
            dp[at(i, j)] = cost;
            from[at(i, j)] = how;
        }
    }
    let mut end_i = 0;
    for i in 1..=n {
        if dp[at(i, m)] < dp[at(end_i, m)] {
            end_i = i;
        }
    }
    // Backtrack. `vote[spoken word][.]` collects original words its matched chars landed on.
    let mut matched: Vec<Option<u32>> = vec![None; m];
    let (mut i, mut j) = (end_i, m);
    while i > 0 || j > 0 {
        let how = if i == 0 {
            2
        } else if j == 0 {
            1
        } else {
            from[at(i, j)]
        };
        match how {
            0 => {
                i -= 1;
                j -= 1;
                if orig[i].ch == spoken[j].ch {
                    matched[j] = Some(orig[i].word);
                }
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    let mut origins = vec![None; n_spoken_words];
    let mut counts: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n_spoken_words];
    for (item, word) in spoken.iter().zip(&matched) {
        if let Some(word) = word {
            let slot = &mut counts[item.word as usize];
            if let Some(pair) = slot.iter_mut().find(|pair| pair.0 == *word) {
                pair.1 += 1;
            } else {
                slot.push((*word, 1));
            }
        }
    }
    let mut previous = None;
    for (index, votes) in counts.iter().enumerate() {
        let best = votes.iter().max_by_key(|pair| pair.1).map(|pair| pair.0).or(previous);
        origins[index] = best;
        if best.is_some() {
            previous = best;
        }
    }
    // A spoken word with no matched letter, such as "dollars" read out of "$2", keeps the
    // previous word. Leading ones take the next word that did match.
    let mut following = None;
    for origin in origins.iter_mut().rev() {
        if origin.is_none() {
            *origin = following;
        }
        if origin.is_some() {
            following = *origin;
        }
    }
    (origins, end_i)
}

/// Byte offset in `chunk` for each byte of `prepared`, which is [`prepare_text_prompt`] of it.
///
/// The two differ by the case of the first letter and a closing mark `prepared` may gain.
/// Bytes `prepared` added map to `None`.
fn prepared_to_chunk(chunk: &str, prepared: &str) -> Vec<Option<usize>> {
    let mut map = vec![None; prepared.len()];
    let mut chunk_chars = chunk.char_indices().peekable();
    let mut prepared_at = 0usize;
    if let Some(&(chunk_byte, chunk_char)) = chunk_chars.peek() {
        let upper: String = chunk_char.to_uppercase().collect();
        let mut buf = [0u8; 4];
        let as_written = chunk_char.encode_utf8(&mut buf).to_string();
        let take = if prepared[prepared_at..].starts_with(&upper) {
            Some(upper.len())
        } else if prepared[prepared_at..].starts_with(&as_written) {
            Some(as_written.len())
        } else {
            None
        };
        if let Some(take) = take {
            map[prepared_at..prepared_at + take].fill(Some(chunk_byte));
            prepared_at += take;
            chunk_chars.next();
        }
    }
    while prepared_at < prepared.len() {
        let Some(&(chunk_byte, chunk_char)) = chunk_chars.peek() else { break };
        let prepared_char = prepared[prepared_at..].chars().next().unwrap();
        if prepared_char != chunk_char {
            break;
        }
        let width = prepared_char.len_utf8();
        map[prepared_at..prepared_at + width].fill(Some(chunk_byte));
        prepared_at += width;
        chunk_chars.next();
    }
    map
}

fn even_spans(n: usize, text: &str) -> Vec<(usize, usize)> {
    let bytes = text.len();
    (0..n)
        .map(|i| {
            let start = floor_boundary(text, bytes * i / n.max(1));
            let end = floor_boundary(text, bytes * (i + 1) / n.max(1)).max(start);
            (start, end)
        })
        .collect()
}

fn floor_boundary(text: &str, mut index: usize) -> usize {
    if index > text.len() {
        return text.len();
    }
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn word_index(words: &[(usize, usize)], byte: usize) -> Option<usize> {
    let index = words.partition_point(|&(start, _)| start <= byte);
    let index = index.checked_sub(1)?;
    (byte < words[index].1).then_some(index)
}

fn whitespace_words(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = start.take() {
                out.push((start, index));
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(start) = start {
        out.push((start, text.len()));
    }
    out
}

/// Whitespace words as [`OrigWord`]s, so [`content_of`] can name them. The char offsets are
/// unused here; the word index is the position in this vec.
fn spoken_spans(text: &str) -> Vec<OrigWord> {
    whitespace_words(text)
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| OrigWord {
            word: text[start..end].to_string(),
            start_byte: start,
            end_byte: end,
            start: index,
            end: index + 1,
        })
        .collect()
}

fn fold_char(ch: char) -> Option<char> {
    ch.to_lowercase().next().filter(|ch| ch.is_alphanumeric())
}

/// The words of `text` a caller would highlight: letter and number runs, with an internal
/// hyphen, dot, underscore or apostrophe kept when a letter or number follows it.
fn words_in(text: &str) -> Vec<OrigWord> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if !chars[index].1.is_alphanumeric() {
            index += 1;
            continue;
        }
        let mut start_byte = chars[index].0;
        index += 1;
        while index < chars.len() {
            if chars[index].1.is_alphanumeric() {
                index += 1;
                continue;
            }
            let internal = matches!(chars[index].1, '-' | '.' | '_' | '\'' | '\u{2019}');
            if internal && chars.get(index + 1).is_some_and(|(_, ch)| ch.is_alphanumeric()) {
                index += 1;
                continue;
            }
            break;
        }
        let end_byte = chars.get(index).map(|(byte, _)| *byte).unwrap_or(text.len());
        // A currency sign glued to the front is part of the word the caller wants highlighted.
        if let Some((sym_at, sym)) = text[..start_byte].char_indices().next_back()
            && matches!(sym, '$' | '£' | '€' | '¥' | '₹' | '₩' | '#')
            && sym_at + sym.len_utf8() == start_byte
        {
            start_byte = sym_at;
        }
        let start = text[..start_byte].chars().count();
        let word = text[start_byte..end_byte].to_string();
        let end = start + word.chars().count();
        out.push(OrigWord { word, start_byte, end_byte, start, end });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::{Lang, Normalize};

    fn row(weights: &[f32]) -> Vec<f32> {
        weights.to_vec()
    }

    #[test]
    fn the_token_walk_advances_only_when_the_next_token_wins() {
        assert_eq!(step_token(0, &[]), 0);
        assert_eq!(step_token(0, &[0.9, 0.1, 0.0]), 0);
        assert_eq!(step_token(0, &[0.4, 0.6, 0.0]), 1);
        // Two tokens ahead does not skip: only the neighbour is eligible.
        assert_eq!(step_token(0, &[0.1, 0.2, 0.7]), 1);
        assert_eq!(step_token(1, &[0.1, 0.2, 0.7]), 2);
        assert_eq!(step_token(2, &[0.0, 0.0, 1.0]), 2);
    }

    #[test]
    fn a_clock_times_words_from_the_rows_and_closes_the_chunk() {
        let words = ["one", "two", "three"]
            .into_iter()
            .enumerate()
            .map(|(i, word)| OrigWord {
                word: word.to_string(),
                start_byte: 0,
                end_byte: 0,
                start: i * 4,
                end: i * 4 + word.len(),
            })
            .collect();
        let mut clock = WordClock {
            words,
            token_words: vec![vec![Some(0), Some(1), Some(2), None]],
            chunk: 0,
            token_word: Vec::new(),
            frame: 0,
            frame_rate: 12.5,
            prev_token: 0,
            open: None,
            cursor: 0,
        };
        clock.begin_chunk();
        // Stay on "one", step to "two", step to "three", then the added period.
        let mut seen = Vec::new();
        seen.extend(clock.push(&row(&[0.8, 0.2, 0.0, 0.0])));
        seen.extend(clock.push(&row(&[0.7, 0.3, 0.0, 0.0])));
        seen.extend(clock.push(&row(&[0.2, 0.8, 0.0, 0.0])));
        seen.extend(clock.push(&row(&[0.0, 0.2, 0.8, 0.0])));
        seen.extend(clock.push(&row(&[0.0, 0.0, 0.4, 0.6])));
        assert!(seen.iter().all(|word| word.word != "three"), "the last word stays open: {seen:?}");
        seen.extend(clock.end_chunk());
        let timed: Vec<_> =
            seen.iter().map(|word| (word.word.as_str(), word.from, word.to)).collect();
        assert_eq!(timed, [("one", 0.0, 0.16), ("two", 0.16, 0.24), ("three", 0.24, 0.4)]);
    }

    #[test]
    fn words_keep_internal_hyphens_and_apostrophes_and_their_offsets() {
        let text = "Task activity, the test-run and the composer's question.";
        let words = words_in(text);
        let got: Vec<_> = words.iter().map(|word| word.word.as_str()).collect();
        assert_eq!(
            got,
            ["Task", "activity", "the", "test-run", "and", "the", "composer's", "question"]
        );
        for word in &words {
            assert_eq!(&text[word.start_byte..word.end_byte], word.word);
            let slice: String = text.chars().skip(word.start).take(word.end - word.start).collect();
            assert_eq!(slice, word.word);
        }
    }

    struct Words;

    impl Tokenizer for Words {
        fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
            Ok(self.encode_spans(text)?.into_iter().map(|(id, _, _)| id).collect())
        }

        fn decode(&self, ids: &[u32]) -> xn::Result<String> {
            Ok(ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "))
        }

        fn encode_spans(&self, text: &str) -> xn::Result<Vec<(u32, usize, usize)>> {
            Ok(whitespace_words(text)
                .into_iter()
                .enumerate()
                .map(|(index, (start, end))| ((index + 1) as u32, start, end))
                .collect())
        }
    }

    fn chunk_of(text: &str, normalize: Normalize) -> Chunk {
        // The chunk is the normalized text, which is what a one-sentence request becomes
        // before `prepare_text_prompt`. Built directly so the test tokenizer does not have
        // to round-trip through `split_into_best_sentences`, whose decode is the model's.
        Chunk::new(normalize.apply(text).into_owned(), &Words, 12.5).unwrap()
    }

    #[test]
    fn the_example_sentence_maps_every_word_back_onto_itself() {
        let text = "Task activity, the workflow graph, tabs, test-run controls and the composer's question then mixed the two records up.";
        let normalize = Normalize::for_lang(Lang::En);
        let chunks = vec![chunk_of(text, normalize)];
        let map = map_utterance(text, &chunks, &Words).unwrap();
        let words: Vec<_> = map.words().collect();
        assert_eq!(
            words,
            [
                "Task",
                "activity",
                "the",
                "workflow",
                "graph",
                "tabs",
                "test-run",
                "controls",
                "and",
                "the",
                "composer's",
                "question",
                "then",
                "mixed",
                "the",
                "two",
                "records",
                "up"
            ]
        );
        // Every token that is a word of the chunk resolves to one of those words.
        assert!(map.token_words.iter().flatten().any(Option::is_some));
        let mut clock = map.clock(12.5);
        let mut timings = Vec::new();
        for chunk in &map.token_words {
            clock.begin_chunk();
            // One frame per token, each row peaked on the next token, so the walk visits all.
            for index in 0..chunk.len() {
                let mut weights = vec![0.0; chunk.len()];
                weights[index] = 1.0;
                if index > 0 {
                    weights[index - 1] = 0.1;
                }
                timings.extend(clock.push(&weights));
            }
            timings.extend(clock.end_chunk());
        }
        let spoken: Vec<_> = timings.iter().map(|word| word.word.as_str()).collect();
        assert_eq!(spoken, words);
        assert!(timings.windows(2).all(|pair| pair[0].to <= pair[1].from));
        assert!(timings.iter().all(|word| word.to >= word.from));
        assert_eq!(timings[0].start, 0);
        assert_eq!(timings[0].word, "Task");
    }

    #[test]
    fn a_rewrite_that_grows_a_word_still_points_at_the_original() {
        let text = "It costs $2 today.";
        let normalize = Normalize::for_lang(Lang::En);
        let normalized = normalize.apply(text);
        assert!(
            normalized.split_whitespace().count() > text.split_whitespace().count(),
            "{normalized}"
        );
        let chunks = vec![chunk_of(text, normalize)];
        let map = map_utterance(text, &chunks, &Words).unwrap();
        let words: Vec<_> = map.words().collect();
        assert_eq!(words, ["It", "costs", "$2", "today"]);
        let amount = map.words.iter().find(|word| word.word == "$2").unwrap();
        assert_eq!(&text[amount.start_byte..amount.end_byte], "$2");
        let mut clock = map.clock(12.5);
        clock.begin_chunk();
        let n = map.token_words[0].len();
        let mut timings = Vec::new();
        for index in 0..n {
            let mut weights = vec![0.0; n];
            weights[index] = 1.0;
            timings.extend(clock.push(&weights));
        }
        timings.extend(clock.end_chunk());
        // The spoken expansion ("2 dollars") highlights the original "2", not a new word.
        assert!(timings.iter().any(|word| word.word == "$2"));
        assert!(timings.iter().all(|word| words.contains(&word.word.as_str())));
    }
}
