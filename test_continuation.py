"""Alien-grab recovery: bounded retries and model-only executable calls."""
import json
import unittest
from unittest.mock import Mock, patch

import opencode_proxy as proxy


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, turns, mode="auto", offered=None, forced=None,
                       stream=False, strikes=0, clock=None):
        offered = ["power---file_read"] if offered is None else offered
        prompts, affinities, streamed, logs = [], [], [], []
        key = "continuation-test"
        proxy._ALIEN_STRIKES.clear()
        for _ in range(strikes):
            proxy._strike_add(key)

        async def events(model, prompt, **kwargs):
            prompts.append(prompt)
            affinities.append(kwargs.get("affinity"))
            turn = turns[len(prompts) - 1]
            for item in turn:
                if isinstance(item, Exception):
                    raise item
                yield "text", item

        async def on_text(text):
            streamed.append(text)

        with patch.object(proxy, "acp_turn_events", events), \
                patch.object(proxy, "_drop_if_idle"):
            if clock:
                timer = patch.object(proxy, "time",
                                     Mock(wraps=proxy.time, monotonic=clock))
            else:
                timer = patch.object(proxy.time, "monotonic", proxy.time.monotonic)
            with timer:
                result = await proxy.collect(
                    "test", "[user]\nRead /requested/file", log=logs.append,
                    stream_cb=on_text if stream else None,
                    affinity=("session", "", 1, None, "hash"),
                    enforce=(offered, forced, mode), strike_key=key)
        return result, prompts, affinities, streamed, logs

    def alien(self):
        return proxy.AcpError("alien_tool_frame:read:blocked")

    def fence(self, name="power---file_read"):
        return '<tool_call>' + json.dumps({
            "name": name, "arguments": {"path": "/requested/file"}
        }) + '</tool_call>'

    async def test_model_fence_recovers_block_and_stream(self):
        for streaming in (False, True):
            result, prompts, affinities, streamed, logs = await self.run_turn(
                [["Checking file. ", self.alien()], [self.fence()]], stream=streaming)
            text, calls, alien, attempts = result
            self.assertEqual(attempts, 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["function"]["name"], "power---file_read")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                             {"path": "/requested/file"})
            self.assertIsNone(affinities[1])
            self.assertIn("Continue with a fenced call.", prompts[1])
            self.assertNotIn('[tool result:', prompts[1])
            self.assertNotIn("blocked native", "".join(streamed))
            self.assertTrue(any("alien correction" in log for log in logs))
            body = proxy.final_body("id", "test", text, calls)
            self.assertEqual(len(body["choices"]), 1)
            self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
            self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
            self.assertNotIn("continuation-test", proxy._ALIEN_STRIKES)

    async def test_repeated_grab_returns_guidance_without_call(self):
        result, prompts, _, streamed, _ = await self.run_turn(
            [[self.alien()], [self.alien()]], stream=True)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(result[1], [])
        self.assertIn("In the next model turn", result[0])
        self.assertEqual("".join(streamed), result[0])

    async def test_third_strike_still_fails(self):
        with self.assertRaisesRegex(proxy.AcpError, "refusing to loop"):
            await self.run_turn([[self.alien()]], strikes=2)

    async def test_no_tools_and_none_never_retry_or_call(self):
        for offered, mode in [([], "auto"), (["power---file_read"], "none")]:
            result, prompts, *_ = await self.run_turn(
                [["No tool output.", self.alien()]], offered=offered, mode=mode)
            self.assertEqual(len(prompts), 1)
            self.assertEqual(result[1], [])

    async def test_unoffered_fence_is_not_executable(self):
        result, *_ = await self.run_turn([[self.alien()], [self.fence("power---bash")]])
        self.assertEqual(result[1], [])

    async def test_prose_recovery_does_not_fabricate_call(self):
        result, *_ = await self.run_turn([[self.alien()], ["Cannot read that file."]])
        self.assertEqual(result[1], [])
        self.assertEqual(result[0], "Cannot read that file.")

    async def test_forced_tool_still_rejects_wrong_model_call(self):
        with self.assertRaisesRegex(proxy.ShimError, "produced no call"):
            await self.run_turn([[self.alien()], [self.fence("power---bash")]],
                                mode="required", forced="power---file_read")

    async def test_low_budget_returns_guidance_without_retry(self):
        times = iter([0, 0, 0, 1490, 1490])
        result, prompts, *_ = await self.run_turn(
            [[self.alien()]], clock=lambda: next(times, 1490))
        self.assertEqual(len(prompts), 1)
        self.assertEqual(result[1], [])
        self.assertIn("blocked native", result[0])



if __name__ == "__main__":
    unittest.main()
