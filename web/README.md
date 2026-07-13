# ClutchCull Web (Next.js frontend)

Premium front-end for ClutchCull. Calls the FastAPI backend (`../api`) for
culling and canvas. Deploys to Vercel; the design target is
`prototype/index.html`.

## Dev
```bash
npm install
npm run dev   # http://localhost:3000
```
Set `NEXT_PUBLIC_API_URL` (see `.env.example`) to the ClutchCull API Space.
