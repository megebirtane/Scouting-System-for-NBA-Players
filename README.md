# Nerede İzlenir

🚀 Live Demo: [Check out the interactive app here](https://nerede-api.onrender.com/)
Turkish streaming discovery platform - Find where movies and TV shows are available across Turkish streaming platforms.

![Platform](https://img.shields.io/badge/platform-web-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

## Features

- **Web Interface**: Beautiful, cinematic website for searching and discovering content
- **Search**: Find movies and TV shows with Turkish titles and descriptions
- **Provider Discovery**: See which Turkish streaming platforms have specific content
- **ML Recommendations**: Get personalized content recommendations based on your preferences
- **User Accounts**: Track your watch history and streaming subscriptions
- **Caching**: Redis-powered caching for fast responses
- **Mobile Responsive**: Fully optimized for mobile devices and tablets

## Supported Streaming Platforms

- Netflix
- BluTV
- Disney+
- Amazon Prime Video
- Mubi
- TOD
- Apple TV+

## Tech Stack

- **Backend**: FastAPI (async Python)
- **Frontend**: Vanilla JS with cinematic dark theme
- **Database**: PostgreSQL with async SQLAlchemy
- **Cache**: Redis
- **ML**: scikit-learn (TF-IDF, cosine similarity)
- **External API**: TMDB (The Movie Database)
- **Deployment**: Docker, Render

## Quick Start

### Prerequisites

- Docker and Docker Compose
- TMDB API key (get one at https://www.themoviedb.org/settings/api)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mertcan-tas/nerede-izlenir.git
cd nerede-izlenir
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

3. Start the services:
```bash
docker-compose up -d
```

4. Seed the providers:
```bash
curl -X POST http://localhost:8000/api/v1/providers/seed
```

5. Access the application:
- **Website**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## TMDB API Configuration

The application supports both TMDB API versions:

- **v3 API Key**: Standard API key (recommended)
- **v4 JWT Token**: Read access token for advanced features

Set your API key in the `.env` file:
```bash
TMDB_API_KEY=your_api_key_here
```

The application automatically detects whether you're using a v3 API key or v4 JWT token.

## Deployment

### Render (Recommended)

The project is configured for easy deployment on Render:

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set the following environment variables:
   - `TMDB_API_KEY` - Your TMDB API key
   - `DATABASE_URL` - PostgreSQL connection URL
   - `REDIS_URL` - Redis connection URL
   - `SECRET_KEY` - A secure secret key for JWT

The `render.yaml` file contains the deployment configuration.

### Docker

```bash
docker-compose up -d
```

## Frontend

The website is a single-page application with a cinematic, editorial design. It features:

- Dark theme with gold accents
- Film-inspired animations and textures
- **Fully responsive design** for mobile, tablet, and desktop
- Real-time search with provider information
- User authentication (login/register)
- Content detail modals with streaming availability

### Frontend Files

Located in the `frontend/` directory:
- `index.html` - Main HTML structure
- `styles.css` - Cinematic theme styles with mobile-first approach
- `app.js` - Application logic and API integration

The frontend is automatically served by FastAPI at the root URL (`/`).

## API Endpoints

### Search
- `GET /api/v1/search?query={query}` - Search for movies and TV shows

### Recommendations
- `GET /api/v1/recommendations?content_id={id}&content_type={movie|tv}` - Get recommendations
- `GET /api/v1/recommendations/personalized` - Get personalized recommendations (auth required)

### Providers
- `GET /api/v1/providers` - List Turkish streaming providers

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `POST /api/v1/auth/refresh` - Refresh access token

### User
- `GET /api/v1/users/me` - Get current user profile
- `GET /api/v1/users/me/subscriptions` - Get user's streaming subscriptions
- `PUT /api/v1/users/me/subscriptions` - Update subscriptions
- `POST /api/v1/users/me/history` - Add to watch history
- `GET /api/v1/users/me/history` - Get watch history

### Health
- `GET /api/v1/health` - Health check for all services

## Development

### Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run with auto-reload:
```bash
uvicorn app.main:app --reload
```

### Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=app --cov-report=html
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TMDB_API_KEY` | TMDB API key or v4 JWT token (required) | - |
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://nerede:nerede@localhost:5432/nerede_izlenir` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `SECRET_KEY` | JWT secret key | `dev-secret-key...` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT access token expiry | `30` |
| `DEBUG` | Enable debug mode | `false` |

## Project Structure

```
nerede-izlenir/
├── app/
│   ├── api/v1/          # API endpoints and schemas
│   ├── core/            # Core utilities (security, exceptions)
│   ├── db/models/       # SQLAlchemy models
│   ├── services/        # Business logic services
│   ├── ml/              # ML recommendation engine
│   └── utils/           # Helper utilities
├── frontend/            # Static frontend files
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── render.yaml          # Render deployment config
├── docker-compose.yml   # Docker configuration
└── requirements.txt     # Python dependencies
```

## Recent Updates

- **Mobile Responsiveness**: Improved UI/UX for mobile devices with responsive layouts
- **TMDB API Flexibility**: Support for both v3 API key and v4 JWT token
- **Render Deployment**: Added configuration for easy cloud deployment
- **Bug Fixes**: Various fixes for deployment compatibility and API handling

## License

MIT
