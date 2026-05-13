# Contributing

These notes are field observations from one team running a specific stack on
specific hardware. We expect them to age — JetPack updates, llama.cpp commits,
NeMo versions, PyTorch fixes, all of these can change what's true.

If you have:

- **A reproduction of a finding on a different stack** (different JetPack,
  different model, different Jetson SKU) — open an issue or PR with your
  numbers. Cross-reproductions are the most valuable thing.
- **A fix or workaround** we missed — especially for the PyTorch NVML
  conflict in [docs/pytorch-nvml-conflict.md](docs/pytorch-nvml-conflict.md) —
  please open a PR. Even pointing at an upstream issue or merged fix is
  enough.
- **Updated numbers from a newer llama.cpp commit or JetPack version** —
  PR welcome. Append your numbers to the relevant doc; don't replace ours
  (the older data is useful for showing the trajectory).
- **A correction** — if something is plain wrong, file an issue with the
  evidence and we'll fix it. We tried to be honest but we wrote this in the
  middle of building a real product so omissions are likely.

We are not going to:

- Maintain this as a comprehensive Jetson edge-LLM resource. It's a focused
  log of what we learned while shipping. If it grows beyond that scope, we'll
  probably split it.
- Run benchmarks on hardware we don't own. We can't validate Orin NX or AGX
  Orin findings — but we'll gladly merge PRs from someone who can.
- Provide commercial support. If you need that, hire someone.

## License

By contributing you agree your contribution is licensed under the same
[Apache License 2.0](LICENSE) as the rest of the repo.
