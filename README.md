# Diet AI Mobile App

Expo React Native mobile application for tracking meals and nutrition using AI-powered food recognition.

## Prerequisites

- Node.js (v18 or later)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- Python 3.8+ (for backend)
- Firebase project with Authentication and Firestore enabled

## Setup

### 1. Install Dependencies

```bash
cd mobile
npm install
```

### 2. Environment Variables

Create a `.env` file in the `mobile` directory with the following variables:

```env
# Firebase Configuration
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_auth_domain
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_storage_bucket
FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
FIREBASE_APP_ID=your_app_id

# Backend API Base URL
# For Android Emulator: use 10.0.2.2 (maps to localhost on host machine)
# For iOS Simulator: use localhost or your machine's IP
# For Physical Device: use your machine's IP address
EXPO_PUBLIC_API_BASE_URL=http://10.0.2.2:8000
```

**Important Notes:**
- `10.0.2.2` is a special IP address that maps to `localhost` (127.0.0.1) on the host machine when running Android emulator
- For iOS Simulator, you can use `http://localhost:8000`
- For physical devices, use your machine's local IP address (e.g., `http://192.168.1.100:8000`)

### 3. Backend Setup

The mobile app requires the FastAPI backend to be running for photo-based meal prediction.

#### Start Backend Server

```bash
# From the project root
cd backend

# Activate virtual environment (if using one)
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend should be accessible at `http://localhost:8000` (or `http://0.0.0.0:8000`).

### 4. Run the Mobile App

```bash
# From the mobile directory
cd mobile
npx expo start -c
```

The `-c` flag clears the cache, which is recommended after environment variable changes.

### 5. Run on Device/Emulator

- **Android Emulator**: Press `a` in the Expo terminal or scan the QR code with Expo Go app
- **iOS Simulator**: Press `i` in the Expo terminal (macOS only)
- **Physical Device**: Scan the QR code with Expo Go app (iOS) or Camera app (Android)

## Project Structure

```
mobile/
├── src/
│   ├── assets/
│   │   └── data/
│   │       └── foods.json          # Local food database
│   ├── components/                 # Reusable UI components
│   ├── navigation/                 # Navigation setup
│   ├── screens/                    # Screen components
│   ├── services/                   # API and service layer
│   │   ├── authService.ts          # Firebase Auth
│   │   ├── db.ts                   # Firestore operations
│   │   ├── firebase.ts             # Firebase initialization
│   │   ├── foods.ts                 # Local food data service
│   │   └── predict.ts              # Backend prediction API
│   ├── store/                      # State management (Zustand)
│   └── utils/                      # Utilities and theme
├── app.config.js                   # Expo configuration
├── .env                            # Environment variables (not in git)
└── package.json
```

## Features

- **Authentication**: Firebase Auth with email/password
- **Onboarding**: User profile setup with calorie goal calculation
- **Manual Meal Logging**: Search and add meals from local database
- **AI Meal Logging**: Take/select photo to predict food using backend ML model
- **Daily Summary**: View calories, macros, and remaining calories
- **Meal Management**: View and delete logged meals
- **Exercise Tracking**: Log exercises and track calories burned
- **Date Navigation**: Switch between Today and Yesterday
- **Bilingual Support**: Turkish (TR) and English (EN) language switching
- **Settings**: Change language, view account info, and logout

## Troubleshooting

### Backend Connection Issues

- **Android Emulator**: Ensure `EXPO_PUBLIC_API_BASE_URL=http://10.0.2.2:8000`
- **iOS Simulator**: Use `http://localhost:8000` or your machine's IP
- **Physical Device**: Use your machine's local IP address
- Verify backend is running: `curl http://localhost:8000/`

### Firebase Errors

- Ensure all Firebase environment variables are set correctly
- Check Firebase project settings in Firebase Console
- Verify Firestore rules allow authenticated users to read/write

### Prediction Timeout

- Ensure backend is running and accessible
- Check network connectivity
- Verify backend URL in `.env` matches your setup
- Prediction has a 20-second timeout

## Development

### Clear Cache

```bash
npx expo start -c
```

### Reset App Data

- Uninstall and reinstall the app
- Or clear AsyncStorage data in development

## Language Settings

The app supports Turkish (TR) and English (EN) languages. 

- Default language is determined by:
  1. Saved language preference (from previous session)
  2. Device locale (if Turkish or English)
  3. Falls back to Turkish if neither is available

- To change language:
  1. Navigate to Settings tab
  2. Select your preferred language (TR/EN)
  3. All screens will update immediately
  4. Language preference persists across app restarts

## Notes

- The app uses local `foods.json` for offline food search
- Meal logs are stored in Firestore under `users/{uid}/logs/{dateKey}/meals/{mealId}`
- Exercise logs are stored in Firestore under `users/{uid}/logs/{dateKey}/exercises/{exerciseId}`
- Photo predictions require backend to be running
- Auth state persists across app restarts using AsyncStorage
- Language preference is saved in AsyncStorage and persists across sessions

