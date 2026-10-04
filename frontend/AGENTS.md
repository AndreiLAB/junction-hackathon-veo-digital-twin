<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Architecture rules

- The app is the AutoTag360 frontend specified in the team repo's `frontend/LOVABLE_PROMPT.md`; that file and the FastAPI backend are the source of truth for screens and response shapes.
- `src/lib/api.ts` is the only data seam: all data comes from the REST backend, no mock data on the working path; unknown values render as "—".
- The API base URL lives in React context (`ApiContext`), defaults to `VITE_API_BASE_URL`, and is persisted to localStorage in an effect (never read during render) to avoid hydration mismatches.
- API links are relative; always prefix them with `abs(base, path)`.
- Photo overlay boxes are in `frame_px` photo coordinates; the SVG overlay uses a `viewBox` in those units so no manual scaling is needed. Expected (geometry) boxes are dashed, detected boxes solid.
- Box/status colours are CSS variables in `src/styles.css`, never hardcoded in components.
