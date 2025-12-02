/**
 * Firebase Functions backend for LunaCare Doctor Portal
 */

const functions = require("firebase-functions");
const admin = require("firebase-admin");
const express = require("express");
const cors = require("cors");
const axios = require("axios");

admin.initializeApp();
const db = admin.firestore();

const app = express();

// Allow requests from your frontend; in production you can restrict origin
app.use(cors({ origin: true }));
app.use(express.json());

// =========================
// Cloud Run ML service
// =========================

// IMPORTANT: keep the /predict at the end
const ML_URL =
    "https://luna-ml-service-824073129263.us-central1.run.app/predict";

// Helper to call ML service on Cloud Run
async function callMLService(features) {
    const response = await axios.post(ML_URL, { features });
    return response.data; // { class, probabilities, feature_order }
}

// Convert RF output to a nicer risk object for the UI
function mapMLToRisk(mlResult) {
    if (!mlResult) {
        return { class: null, label: "UNKNOWN", percent: null, probabilities: null };
    }

    const klass = mlResult.class;
    let label = "STABLE";
    if (klass === 1) label = "MEDIUM";
    else if (klass === 2) label = "HIGH RISK";

    let percent = null;
    if (Array.isArray(mlResult.probabilities)) {
        const p = mlResult.probabilities[klass] ?? null;
        if (p != null) {
            percent = Math.round(p * 100);
        }
    }

    return {
        class: klass,
        label,
        percent,
        probabilities: mlResult.probabilities ?? null,
    };
}

// =========================
// Firestore helpers
// =========================

/**
 * Helper: get latest document from a subcollection ordered by `date`.
 */
async function getLatestLog(userRef, subcollection) {
    const snap = await userRef
        .collection(subcollection)
        .orderBy("date", "desc")
        .limit(1)
        .get();

    if (snap.empty) return null;
    const doc = snap.docs[0];
    return { id: doc.id, ...doc.data() };
}

/**
 * Helper: get a "lastActive" timestamp for dashboard.
 * We look at latest mood_log or apple_watch_log date and return the max.
 */
async function getLastActive(userRef) {
    const [moodLog, watchLog] = await Promise.all([
        getLatestLog(userRef, "mood_logs"),
        getLatestLog(userRef, "apple_watch_logs"),
    ]);

    const dates = [];
    if (moodLog && moodLog.date) dates.push(moodLog.date.toDate());
    if (watchLog && watchLog.date) dates.push(watchLog.date.toDate());

    if (dates.length === 0) return null;

    // latest date
    return new Date(Math.max(...dates.map((d) => d.getTime())));
}

/**
 * Build the feature vector for the ML model from Firestore logs.
 * We use ALL three collections:
 *   - mood_logs
 *   - symptom_logs
 *   - apple_watch_logs
 *
 * The feature names must match ppd_rf_feature_order.json
 */
async function buildFeaturesForUser(userId) {
    const userRef = db.collection("user-test").doc(userId);

    const [moodSnap, symptomSnap, watchSnap] = await Promise.all([
        userRef.collection("mood_logs").orderBy("date", "desc").limit(1).get(),
        userRef.collection("symptom_logs").orderBy("date", "desc").limit(1).get(),
        userRef.collection("apple_watch_logs").orderBy("date", "desc").limit(1).get(),
    ]);

    if (moodSnap.empty || symptomSnap.empty || watchSnap.empty) {
        throw new Error(
            `Missing mood / symptom / apple_watch logs for user ${userId}`
        );
    }

    const moodDoc = moodSnap.docs[0].data();
    const symptomDoc = symptomSnap.docs[0].data();
    const watchDoc = watchSnap.docs[0].data();

    const sleep = watchDoc.sleep_metrics || {};
    const heart = watchDoc.heart_metrics || {};
    const activity = watchDoc.activity_metrics || {};
    const values = symptomDoc.values || {};

    // IMPORTANT: these keys should line up with FEATURE_ORDER
    // in ppd_rf_feature_order.json (snake_case).
    const features = {
        // mood_logs
        mood: typeof moodDoc.mood === "number" ? moodDoc.mood : 3,

        // apple_watch_logs
        sleep_hours:
            typeof sleep.sleep_hours === "number" ? sleep.sleep_hours : 0,
        hrv: typeof heart.hrv === "number" ? heart.hrv : 0,
        resting_hr:
            typeof heart.resting_heart_rate === "number"
                ? heart.resting_heart_rate
                : 0,
        steps: typeof activity.steps === "number" ? activity.steps : 0,
        exercise_minutes:
            typeof activity.exercise_minutes === "number"
                ? activity.exercise_minutes
                : 0,

        // symptom_logs (we keep them as symptoms, NOT "fatigue")
        appetite: Number(values["Appetite"] ?? 0),
        bleeding: Number(values["Bleeding"] ?? 0),
        hair_loss: Number(values["Hair Loss"] ?? 0),
        sleep_trouble: Number(values["Sleep Trouble"] ?? 0),
        // If you added more symptom fields in training, add them here as well.
    };

    return features;
}

// =========================
// Routes
// =========================

/**
 * GET /users
 * Dashboard list:
 *   id, displayName, status, lastActive, riskLabel, riskPercent
 */
app.get("/users", async (req, res) => {
    try {
        const snap = await db
            .collection("user-test")
            .limit(20)
            .get();

        const users = [];
        for (const doc of snap.docs) {
            const data = doc.data();
            const userId = doc.id;
            const userRef = db.collection("user-test").doc(userId);

            let features = null;
            let risk = { class: null, label: "UNKNOWN", percent: null };

            let lastActive = null;

            try {
                [features, lastActive] = await Promise.all([
                    buildFeaturesForUser(userId),
                    getLastActive(userRef),
                ]);

                const mlResult = await callMLService(features);
                risk = mapMLToRisk(mlResult);
            } catch (e) {
                console.warn(
                    `Could not compute ML risk for user ${userId}:`,
                    e.message
                );
            }

            users.push({
                id: userId,
                displayName:
                    data.displayName ||
                    `${data.firstname || ""} ${data.lastname || ""}`,
                email: data.email || null,
                age: data.age || null,
                is_pregnant: data.is_pregnant ?? null,
                created_at: data.created_at || null,
                lastActive,
                riskLabel: risk.label,
                riskPercent: risk.percent,
                riskClass: risk.class,
            });
        }

        res.status(200).json({ users });
    } catch (err) {
        console.error("GET /users error:", err);
        res.status(500).json({ error: "Internal server error" });
    }
});

/**
 * GET /users/:id
 * Detailed view for a single patient:
 *   profile, mood/symptom/watch time series, features + ML risk
 */
app.get("/users/:id", async (req, res) => {
    const userId = req.params.id;

    try {
        const userRef = db.collection("user-test").doc(userId);
        const userDoc = await userRef.get();

        if (!userDoc.exists) {
            return res.status(404).json({ error: "User not found" });
        }

        const user = userDoc.data();

        const [moodSnap, symptomSnap, watchSnap] = await Promise.all([
            userRef.collection("mood_logs").orderBy("date", "asc").get(),
            userRef.collection("symptom_logs").orderBy("date", "asc").get(),
            userRef.collection("apple_watch_logs").orderBy("date", "asc").get(),
        ]);

        const moodSeries = moodSnap.docs.map((d) => {
            const x = d.data();
            return {
                date: x.date,
                mood: x.mood,
            };
        });

        // We keep symptoms as a full symptom time series,
        // NOT renamed to "fatigue".
        const symptomSeries = symptomSnap.docs.map((d) => {
            const x = d.data();
            const values = x.values || {};
            return {
                date: x.date,
                values, // { Appetite, Bleeding, Hair Loss, Sleep Trouble, ... }
            };
        });

        const watchSeries = watchSnap.docs.map((d) => {
            const x = d.data();
            const sleep = x.sleep_metrics || {};
            const heart = x.heart_metrics || {};
            const activity = x.activity_metrics || {};

            return {
                date: x.date,
                sleep_hours: sleep.sleep_hours ?? null,
                hrv: heart.hrv ?? null,
                resting_hr: heart.resting_heart_rate ?? null,
                steps: activity.steps ?? null,
                exercise_minutes: activity.exercise_minutes ?? null,
            };
        });

        let features = null;
        let risk = { class: null, label: "UNKNOWN", percent: null };

        try {
            features = await buildFeaturesForUser(userId);
            const mlResult = await callMLService(features);
            risk = mapMLToRisk(mlResult);
        } catch (e) {
            console.warn(
                `Could not compute detailed ML risk for user ${userId}:`,
                e.message
            );
        }

        res.status(200).json({
            id: userId,
            profile: {
                displayName:
                    user.displayName || `${user.firstname || ""} ${user.lastname || ""}`,
                email: user.email || null,
                age: user.age || null,
                is_pregnant: user.is_pregnant ?? false,
                created_at: user.created_at || null,
            },
            metrics: {
                moodSeries,
                symptomSeries,
                watchSeries,
            },
            features,
            risk,
        });
    } catch (err) {
        console.error("GET /users/:id error:", err);
        res.status(500).json({ error: "Internal server error" });
    }
});

// Export Express app as a single HTTPS function
exports.api = functions.https.onRequest(app);

