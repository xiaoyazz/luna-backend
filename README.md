# LunaCare Backend

Backend service for my capstone project : the LunaCare doctor portal.

## Tech Stack

- Firebase Functions
- Firestore
- Express.js
- Google Cloud Run
- XGBoost

## Features

- Reads patient data from Firestore
- Provides patient data to the doctor portal
- Combines daily health records for ML prediction
- Sends patient health data to the XGBoost ML service
- Returns a PPD risk score

## Recent Backend Updates

The backend was updated from the previous version to support the new LunaCare data structure and XGBoost model.

Main updates:

- Changed the main Firestore collection from `user-test` to `users`
- Updated the backend to use the new Firestore schema
- Replaced the previous Random Forest ML integration with XGBoost
- Changed the ML request from a single `features` object to multiple daily `records`
- Added support for the 28 daily features required by the XGBoost model
- Added mapping from Firestore camelCase fields to the field names expected by the ML model
- Updated date handling to use `createdAt`
- Added support for data from:
  - `measurements`
  - `mood_logs`
  - `symptom_logs`
- Uses up to the latest 30 days of measurement data for ML prediction
- Changed the ML output from class-based prediction to a continuous `risk_score`
- Removed hardcoded Low / Medium / High risk categories
- Missing health values are kept as `null` instead of being replaced with fake values

## Project Structure

```text
luna-backend/
├── functions/
│   ├── index.js
│   ├── package.json
│   └── package-lock.json
├── .firebaserc
├── firebase.json
├── firestore.indexes.json
└── firestore.rules
```

## Setup

Go into the functions folder:

```bash
cd functions
```

Install dependencies:

```bash
npm install
```

## Run Locally

Start the Firebase Functions emulator:

```bash
firebase emulators:start --only functions
```

The API will run at:

```text
http://127.0.0.1:5001/lunacare-d181e/us-central1/api
```

## API Endpoints

### Get All Users

```text
GET /users
```

Example:

```bash
curl http://127.0.0.1:5001/lunacare-d181e/us-central1/api/users
```

### Get One User

```text
GET /users/:id
```

Example:

```bash
curl http://127.0.0.1:5001/lunacare-d181e/us-central1/api/users/USER_ID
```

## Firestore Data

The backend reads patient data from:

```text
users/{userId}
```

Patient data can include:

```text
users/{userId}/measurements
users/{userId}/mood_logs
users/{userId}/symptom_logs
```

### Measurements

The `measurements` collection contains daily health data used for ML prediction, including:

- Mood
- Sleep
- Heart rate
- HRV
- Respiratory rate
- Oxygen saturation
- Activity
- Exercise
- Weight
- Symptoms

### Mood Logs

The `mood_logs` collection stores user mood entries and is also used for mood history in the doctor portal.

### Symptom Logs

The `symptom_logs` collection stores self-reported symptoms such as:

- Fatigue
- Appetite
- Bleeding
- Hair loss
- Sleep trouble

If a matching symptom value is not available for a day, the backend can leave that ML value as `null`.

## ML Integration

The backend sends daily patient health records to the LunaCare XGBoost ML service deployed on Google Cloud Run.

The backend sends data in this format:

```json
{
  "records": [
    {
      "fatigue_1to10": null,
      "mood_1to5": 3,
      "sleep_hours": 6.5,
      "avg_heart_rate_bpm": 82,
      "steps": 4200
    }
  ]
}
```

The ML service processes multiple days of patient data and returns a PPD risk score.

Example response:

```json
{
  "risk_score": 0.57
}
```

The score ranges from 0 to 1.

A higher score represents a higher predicted PPD risk score.

## Notes

The PPD risk score is an ML model output and is intended to support the LunaCare capstone prototype. It should not be treated as a medical diagnosis.
