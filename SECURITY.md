# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting: on this repository, go to the
**Security** tab → **Report a vulnerability**. Reports go directly to the
maintainers, and GitHub keeps the discussion private until a fix is
coordinated.

Include what you can: the affected component (CLI, C API, kernels, app, file
format parsers), a reproduction, and the impact you believe it has. The PLY /
SPZ / COLMAP parsers process untrusted input files and are in scope; so is
the C API surface (`splatx_train_c.h`).

We aim to acknowledge reports within a week. This is a small-team project —
coordinated disclosure timelines are agreed per report rather than promised
up front.

## Scope notes

- Training and rendering run entirely locally; the software makes no network
  connections and ships no telemetry.
- Crashes on malformed model or dataset files are security-relevant
  (parser robustness) and welcome as private reports if you believe they are
  exploitable, or as regular issues otherwise.
