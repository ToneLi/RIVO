import unittest

from deploy_agent import (
    HISTORY_TRUNCATION_NOTICE,
    TRUNCATION_NOTICE,
    _fit_messages_to_input_budget,
    _serialized_message_token_span,
    _truncate_text_tokens,
)


class _CharacterTokenizer:
    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
        return "\n".join(
            f"<{message['role']}>{message.get('content', '')}" for message in messages
        )

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


class HistoryTruncationTests(unittest.TestCase):
    def test_tool_output_truncation_keeps_head_and_tail(self):
        tokenizer = _CharacterTokenizer()
        truncated = _truncate_text_tokens(tokenizer, "abcdefgh", 4)
        self.assertEqual(truncated, "ab" + TRUNCATION_NOTICE + "gh")

    def test_fit_uses_original_tool_output_strategy(self):
        tokenizer = _CharacterTokenizer()
        messages = [
            {"role": "system", "content": "KEEP_SYSTEM"},
            {"role": "user", "content": "KEEP_QUESTION"},
            {"role": "assistant", "content": "KEEP_ASSISTANT"},
            {"role": "tool", "content": "HEAD_" + "o" * 600 + "_TAIL"},
        ]
        prompt, tokens = _fit_messages_to_input_budget(
            messages, [], tokenizer, max_input_tokens=450
        )

        self.assertLessEqual(len(tokens), 450)
        self.assertIn("KEEP_SYSTEM", prompt)
        self.assertIn("KEEP_QUESTION", prompt)
        self.assertIn("KEEP_ASSISTANT", prompt)
        self.assertIn("HEAD_", prompt)
        self.assertIn("_TAIL", prompt)
        self.assertIn(TRUNCATION_NOTICE.strip(), prompt)
        self.assertNotIn(HISTORY_TRUNCATION_NOTICE, prompt)

    def test_retrieval_message_span_maps_into_exact_fitted_prompt(self):
        tokenizer = _CharacterTokenizer()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "retrieved evidence"},
        ]
        _, tokens, metadata = _fit_messages_to_input_budget(
            messages, [], tokenizer, max_input_tokens=1000, return_metadata=True
        )
        span = _serialized_message_token_span(
            metadata, {"start": 2, "end": 4}, [], tokenizer, len(tokens)
        )
        self.assertIsNotNone(span)
        span_text = tokenizer.decode(tokens[span["start"] : span["end"]])
        self.assertIn("tool call", span_text)
        self.assertIn("retrieved evidence", span_text)


if __name__ == "__main__":
    unittest.main()
