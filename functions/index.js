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

app.use(cors({ origin: true }));
app.use(express.json());


// ======================================================
// Configuration
// ======================================================

const USERS_COLLECTION = "users";

/**
 * IMPORTANT
 *
 * If your XGBoost Cloud Run revision is still deployed with:
 *
 *   --no-traffic --tag xgb-test
 *
 * temporarily replace this URL with the xgb-test tagged URL.
 *
 * Once XGBoost receives 100% of normal Cloud Run traffic,
 * this regular service URL can stay here.
 */
const ML_URL =
    process.env.ML_URL ||
    "https://luna-ml-service-824073129263.us-central1.run.app/predict";


// ======================================================
// XGBoost Model Fields
// ======================================================

const MODEL_FIELDS = [
    "fatigue_1to10",
    "mood_1to5",
    "bleeding_1to10",
    "hair_loss_1to10",
    "appetite_issue_1to10",
    "sleep_trouble_1to10",

    "sleep_hours",
    "deep_sleep_hours",
    "rem_sleep_hours",
    "core_sleep_hours",
    "sleep_efficiency_pct",
    "wake_after_sleep_onset_min",

    "avg_heart_rate_bpm",
    "resting_heart_rate_bpm",
    "walking_heart_rate_avg_bpm",
    "hrv_sdnn_ms",
    "respiratory_rate_bpm",
    "oxygen_saturation_pct",
    "vo2max_ml_kg_min",

    "steps",
    "distance_walked_km",
    "flights_climbed",
    "active_energy_kcal",
    "basal_energy_kcal",
    "exercise_minutes",
    "stand_hours",
    "sunlight_hours",

    "weight_kg",
];


// ======================================================
// ML Service
// ======================================================

/**
 * Send multiple days of health records to Cloud Run.
 *
 * New XGBoost request:
 *
 * {
 *   records: [...]
 * }
 *
 * Expected response:
 *
 * {
 *   risk_score: 0.5763,
 *   target: "ppd_risk_score_0to1",
 *   days_used: 30
 * }
 */
async function callMLService(records) {
    const response = await axios.post(ML_URL, {
        records,
    });

    return response.data;
}


/**
 * Convert XGBoost output into something convenient
 * for the Doctor Portal.
 *
 * We do NOT create LOW / MEDIUM / HIGH thresholds here.
 */
function mapMLToRisk(mlResult) {
    const score = Number(mlResult?.risk_score);

    if (!Number.isFinite(score)) {
        return {
            score: null,
            percent: null,
            label: "UNKNOWN",
        };
    }

    return {
        score,
        percent: Math.round(score * 100),
        label: "PPD RISK SCORE",
    };
}


// ======================================================
// General Helpers
// ======================================================

function numberOrNull(value) {
    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return null;
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : null;
}


/**
 * Read a nested value.
 *
 * Example:
 *
 * readPath(data, ["sleep_metrics", "sleep_hours"])
 */
function readPath(object, path) {
    let value = object;

    for (const key of path) {
        if (
            value === null ||
            value === undefined ||
            typeof value !== "object"
        ) {
            return undefined;
        }

        value = value[key];
    }

    return value;
}


/**
 * Try multiple possible field locations and return
 * the first valid numeric value.
 */
function pickNumber(object, paths) {
    for (const path of paths) {
        const value = Array.isArray(path)
            ? readPath(object, path)
            : object?.[path];

        const number = numberOrNull(value);

        if (number !== null) {
            return number;
        }
    }

    return null;
}


/**
 * Convert Firestore Timestamp / Date / string
 * into YYYY-MM-DD.
 */
function getDateKey(value) {
    if (!value) {
        return null;
    }

    let date;

    if (typeof value.toDate === "function") {
        date = value.toDate();
    } else if (value instanceof Date) {
        date = value;
    } else {
        date = new Date(value);
    }

    if (Number.isNaN(date.getTime())) {
        return null;
    }

    return date.toISOString().slice(0, 10);
}


/**
 * Get milliseconds for timestamp comparison.
 */
function getTimeValue(value) {
    if (!value) {
        return 0;
    }

    if (typeof value.toDate === "function") {
        return value.toDate().getTime();
    }

    if (value instanceof Date) {
        return value.getTime();
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return 0;
    }

    return date.getTime();
}


/**
 * Merge nested objects.
 *
 * This helps in case the measurements collection
 * contains several documents for the same day.
 */
function deepMerge(target, source) {
    const result = {
        ...(target || {}),
    };

    for (const [key, value] of Object.entries(source || {})) {
        if (
            value &&
            typeof value === "object" &&
            !Array.isArray(value) &&
            typeof value.toDate !== "function"
        ) {
            result[key] = deepMerge(
                result[key] || {},
                value
            );
        } else {
            result[key] = value;
        }
    }

    return result;
}


// ======================================================
// Mood Mapping
// ======================================================

/**
 * Current LunaCare mood scale:
 *
 * -2 -> 1
 * -1 -> 2
 *  0 -> 3
 *  2 -> 4
 *  4 -> 5
 *
 * XGBoost expects mood_1to5.
 */
function mapMoodTo1to5(rawMood) {
    const mood = Number(rawMood);

    if (!Number.isFinite(mood)) {
        return null;
    }

    if (mood <= -2) {
        return 1;
    }

    if (mood === -1) {
        return 2;
    }

    if (mood === 0) {
        return 3;
    }

    if (mood === 2) {
        return 4;
    }

    if (mood >= 4) {
        return 5;
    }

    return 3;
}


// ======================================================
// Firestore Helpers
// ======================================================

async function getLatestLog(
    userRef,
    subcollection,
    dateField = "createdAt"
) {
    const snap = await userRef
        .collection(subcollection)
        .orderBy(dateField, "desc")
        .limit(1)
        .get();

    if (snap.empty) {
        return null;
    }

    const doc = snap.docs[0];

    return {
        id: doc.id,
        ...doc.data(),
    };
}


/**
 * Latest activity now comes from:
 *
 * - mood_logs
 * - measurements
 */
async function getLastActive(userRef) {
    const [moodLog, measurementLog] =
        await Promise.all([
            getLatestLog(
                userRef,
                "mood_logs",
                "createdAt"
            ),

            getLatestLog(
                userRef,
                "measurements",
                "createdAt"
            ),
        ]);

    const dates = [];

    if (moodLog?.createdAt) {
        dates.push(
            getTimeValue(moodLog.createdAt)
        );
    }

    if (measurementLog?.createdAt) {
        dates.push(
            getTimeValue(measurementLog.createdAt)
        );
    }

    const validDates =
        dates.filter((value) => value > 0);

    if (validDates.length === 0) {
        return null;
    }

    return new Date(
        Math.max(...validDates)
    );
}


/**
 * Dashboard mood sparkline.
 *
 * New Firestore schema uses createdAt instead of date.
 */
async function getMoodTrend(
    userRef,
    limit = 10
) {
    const snap = await userRef
        .collection("mood_logs")
        .orderBy("createdAt", "desc")
        .limit(limit)
        .get();

    if (snap.empty) {
        return [];
    }

    return snap.docs
        .map((doc) => doc.data())
        .reverse()
        .map((data) => (
            typeof data.mood === "number"
                ? data.mood
                : null
        ))
        .filter((value) => value !== null);
}


// ======================================================
// Measurement Mapping
// ======================================================

/**
 * Convert one day's measurement data into the
 * fields expected by XGBoost.
 *
 * This supports:
 *
 * 1. New flat field names, for example:
 *      sleep_hours
 *
 * 2. The nested structure used by the previous
 *    LunaCare backend, for example:
 *      sleep_metrics.sleep_hours
 *
 * Once the exact new measurements schema is confirmed,
 * these fallbacks can be simplified.
 */
function extractMeasurementFeatures(data) {
    return {
        // Self-reported
        fatigue_1to10:
            pickNumber(data, [
                "fatigue1to10",
                "fatigue_1to10",
            ]),

        mood_1to5:
            pickNumber(data, [
                "mood1to5",
                "mood_1to5",
            ]),

        bleeding_1to10:
            pickNumber(data, [
                "bleeding1to10",
                "bleeding_1to10",
            ]),

        hair_loss_1to10:
            pickNumber(data, [
                "hairLoss1to10",
                "hair_loss_1to10",
            ]),

        appetite_issue_1to10:
            pickNumber(data, [
                "appetiteIssue1to10",
                "appetite_issue_1to10",
            ]),

        sleep_trouble_1to10:
            pickNumber(data, [
                "sleepTrouble1to10",
                "sleep_trouble_1to10",
            ]),

        // Sleep
        sleep_hours:
            pickNumber(data, [
                "sleepHours",
                "sleep_hours",
            ]),

        deep_sleep_hours:
            pickNumber(data, [
                "deepSleepHours",
                "deep_sleep_hours",
            ]),

        rem_sleep_hours:
            pickNumber(data, [
                "remSleepHours",
                "rem_sleep_hours",
            ]),

        core_sleep_hours:
            pickNumber(data, [
                "coreSleepHours",
                "core_sleep_hours",
            ]),

        sleep_efficiency_pct:
            pickNumber(data, [
                "sleepEfficiencyPct",
                "sleep_efficiency_pct",
            ]),

        wake_after_sleep_onset_min:
            pickNumber(data, [
                "wakeAfterSleepOnsetMin",
                "wake_after_sleep_onset_min",
            ]),

        // Heart / fitness
        avg_heart_rate_bpm:
            pickNumber(data, [
                "avgHeartRateBpm",
                "avg_heart_rate_bpm",
            ]),

        resting_heart_rate_bpm:
            pickNumber(data, [
                "restingHRBpm",
                "resting_heart_rate_bpm",
            ]),

        walking_heart_rate_avg_bpm:
            pickNumber(data, [
                "walkingHeartRateAvgBpm",
                "walking_heart_rate_avg_bpm",
            ]),

        hrv_sdnn_ms:
            pickNumber(data, [
                "hrvSDNNms",
                "hrv_sdnn_ms",
            ]),

        respiratory_rate_bpm:
            pickNumber(data, [
                "respiratoryRateBpm",
                "respiratory_rate_bpm",
            ]),

        oxygen_saturation_pct:
            pickNumber(data, [
                "oxygenSaturationPct",
                "oxygen_saturation_pct",
            ]),

        vo2max_ml_kg_min:
            pickNumber(data, [
                "vo2Max",
                "vo2max_ml_kg_min",
            ]),

        // Activity
        steps:
            pickNumber(data, [
                "steps",
            ]),

        distance_walked_km:
            pickNumber(data, [
                "distanceWalkedKm",
                "distance_walked_km",
            ]),

        flights_climbed:
            pickNumber(data, [
                "flightsClimbed",
                "flights_climbed",
            ]),

        active_energy_kcal:
            pickNumber(data, [
                "activeEnergyKcal",
                "active_energy_kcal",
            ]),

        basal_energy_kcal:
            pickNumber(data, [
                "basalEnergyKcal",
                "basal_energy_kcal",
            ]),

        exercise_minutes:
            pickNumber(data, [
                "exerciseMinutes",
                "exercise_minutes",
            ]),

        stand_hours:
            pickNumber(data, [
                "standHours",
                "stand_hours",
            ]),

        sunlight_hours:
            pickNumber(data, [
                "sunlightHours",
                "sunlight_hours",
            ]),

        // Body
        weight_kg:
            pickNumber(data, [
                "weightKg",
                "weight_kg",
            ]),
    };
}


// ======================================================
// Build XGBoost Records
// ======================================================

async function buildRecordsForUser(userId) {
    const userRef = db
        .collection(USERS_COLLECTION)
        .doc(userId);

    const [
        userDoc,
        moodSnap,
        symptomSnap,
        measurementSnap,
    ] = await Promise.all([
        userRef.get(),

        userRef
            .collection("mood_logs")
            .orderBy("createdAt", "desc")
            .limit(100)
            .get(),

        userRef
            .collection("symptom_logs")
            .orderBy("createdAt", "desc")
            .limit(100)
            .get(),

        userRef
            .collection("measurements")
            .orderBy("createdAt", "desc")
            .limit(200)
            .get(),
    ]);


    if (!userDoc.exists) {
        throw new Error(
            `User ${userId} does not exist`
        );
    }


    if (
        moodSnap.empty &&
        measurementSnap.empty
    ) {
        throw new Error(
            `No health data found for user ${userId}`
        );
    }


    // ====================================================
    // Mood logs by date
    // ====================================================

    const moodByDate = new Map();


    /**
     * Query is newest -> oldest.
     *
     * Reverse it so that if there are multiple mood
     * entries on the same day, the newest one becomes
     * the final value stored in the map.
     */
    const moodDocs =
        [...moodSnap.docs].reverse();


    for (const doc of moodDocs) {
        const data = doc.data();

        const dateKey =
            getDateKey(
                data.createdAt ??
                data.date
            );

        if (!dateKey) {
            continue;
        }

        moodByDate.set(
            dateKey,
            data
        );
    }

    // ====================================================
    // Symptom logs by date
    // ====================================================

    const symptomByDate = new Map();

    const symptomDocs =
        [...symptomSnap.docs].reverse();

    for (const doc of symptomDocs) {
        const data = doc.data();

        const dateKey =
            getDateKey(
                data.createdAt ??
                data.date
            );

        if (!dateKey) {
            continue;
        }

        symptomByDate.set(
            dateKey,
            data
        );
    }


    // ====================================================
    // Measurements by date
    // ====================================================

    const measurementsByDate =
        new Map();


    const measurementDocs =
        [...measurementSnap.docs].reverse();


    for (const doc of measurementDocs) {
        const data = doc.data();

        const dateKey =
            getDateKey(
                data.createdAt ??
                data.date
            );

        if (!dateKey) {
            continue;
        }


        const current =
            measurementsByDate.get(dateKey) || {};


        /**
         * If multiple measurement documents exist for the
         * same date, combine them.
         */
        measurementsByDate.set(
            dateKey,
            deepMerge(current, data)
        );
    }


    // ====================================================
    // Dates available in either collection
    // ====================================================

    const allDates = [
        ...measurementsByDate.keys(),
    ]
        .sort()
        .slice(-30);


    if (allDates.length === 0) {
        throw new Error(
            `No usable dated health records found for ${userId}`
        );
    }


    // ====================================================
    // Build daily ML records
    // ====================================================

    const records = [];


    for (const date of allDates) {
        const moodData =
            moodByDate.get(date) || {};

        const symptomData =
            symptomByDate.get(date) || {};

        const symptomValues =
            symptomData.values || {};

        const measurementData =
            measurementsByDate.get(date) || {};


        const measurementFeatures =
            extractMeasurementFeatures(
                measurementData
            );


        const record = {
            ...measurementFeatures,

            // Prefer the value already stored in measurements.
            // Fall back to mood_logs if it is missing.
            mood_1to5:
                measurementFeatures.mood_1to5 ??
                mapMoodTo1to5(
                    moodData.mood
                ),

            // Fatigue currently comes from symptom_logs.
            fatigue_1to10:
                numberOrNull(
                    symptomValues["Fatigue"]
                ) ??
                measurementFeatures.fatigue_1to10,

            // Use measurement values first, with symptom_logs
            // available as fallback.
            appetite_issue_1to10:
                measurementFeatures.appetite_issue_1to10 ??
                numberOrNull(
                    symptomValues["Appetite"]
                ),

            bleeding_1to10:
                measurementFeatures.bleeding_1to10 ??
                numberOrNull(
                    symptomValues["Bleeding"]
                ),

            hair_loss_1to10:
                measurementFeatures.hair_loss_1to10 ??
                numberOrNull(
                    symptomValues["Hair Loss"]
                ),

            sleep_trouble_1to10:
                measurementFeatures.sleep_trouble_1to10 ??
                numberOrNull(
                    symptomValues["Sleep Trouble"]
                ),
        };


        /**
         * Guarantee every field expected by the model
         * exists in the JSON object.
         *
         * Missing health values stay null.
         */
        const completeRecord = {};

        for (const field of MODEL_FIELDS) {
            completeRecord[field] =
                record[field] ?? null;
        }


        records.push(
            completeRecord
        );
    }


    return records;
}


// ======================================================
// Dashboard Route
// ======================================================

/**
 * GET /users
 */
app.get(
    "/users",
    async (req, res) => {
        try {
            const snap = await db
                .collection(USERS_COLLECTION)
                .limit(20)
                .get();


            const users = [];


            for (const doc of snap.docs) {
                const data = doc.data();

                const userId = doc.id;

                const userRef = db
                    .collection(USERS_COLLECTION)
                    .doc(userId);


                let risk = {
                    score: null,
                    percent: null,
                    label: "UNKNOWN",
                };


                let lastActive = null;
                let moodTrend = [];


                try {
                    const [
                        records,
                        active,
                        trend,
                    ] = await Promise.all([
                        buildRecordsForUser(userId),

                        getLastActive(userRef),

                        getMoodTrend(
                            userRef,
                            10
                        ),
                    ]);


                    lastActive = active;
                    moodTrend = trend;


                    const mlResult =
                        await callMLService(records);


                    risk =
                        mapMLToRisk(
                            mlResult
                        );

                } catch (error) {
                    console.warn(
                        `Could not compute ML risk for user ${userId}:`,
                        error.response?.data ||
                        error.message
                    );
                }


                users.push({
                    id:
                        userId,

                    displayName:
                        data.displayName ||
                        `${data.firstName || ""} ${data.lastName || ""}`.trim(),

                    email:
                        data.email || null,

                    age:
                        data.age ?? null,

                    isPregnant:
                        data.isPregnant ??
                        data.is_pregnant ??
                        null,

                    createdAt:
                        data.createdAt ??
                        data.created_at ??
                        null,

                    lastActive,

                    riskScore:
                        risk.score,

                    riskPercent:
                        risk.percent,

                    riskLabel:
                        risk.label,

                    moodTrend,
                });
            }


            res.status(200).json({
                users,
            });

        } catch (error) {
            console.error(
                "GET /users error:",
                error
            );

            res.status(500).json({
                error:
                    "Internal server error",
            });
        }
    }
);


// ======================================================
// Patient Detail Route
// ======================================================

/**
 * GET /users/:id
 */
app.get(
    "/users/:id",
    async (req, res) => {
        const userId =
            req.params.id;


        try {
            const userRef = db
                .collection(USERS_COLLECTION)
                .doc(userId);


            const userDoc =
                await userRef.get();


            if (!userDoc.exists) {
                return res
                    .status(404)
                    .json({
                        error:
                            "User not found",
                    });
            }


            const user =
                userDoc.data();


            const [
                moodSnap,
                measurementSnap,
            ] = await Promise.all([
                userRef
                    .collection("mood_logs")
                    .orderBy(
                        "createdAt",
                        "asc"
                    )
                    .get(),

                userRef
                    .collection("measurements")
                    .orderBy(
                        "createdAt",
                        "asc"
                    )
                    .get(),
            ]);


            // ==================================================
            // Mood Series
            // ==================================================

            const moodSeries =
                moodSnap.docs.map(
                    (doc) => {
                        const data =
                            doc.data();

                        return {
                            id:
                                doc.id,

                            createdAt:
                                data.createdAt ??
                                null,

                            mood:
                                data.mood ??
                                null,

                            mood_1to5:
                                mapMoodTo1to5(
                                    data.mood
                                ),

                            notes:
                                data.notes ??
                                null,

                            source:
                                data.source ??
                                null,
                        };
                    }
                );


            // ==================================================
            // Measurement Series
            // ==================================================

            const measurementSeries =
                measurementSnap.docs.map(
                    (doc) => {
                        const data =
                            doc.data();

                        return {
                            id:
                                doc.id,

                            createdAt:
                                data.createdAt ??
                                null,

                            ...extractMeasurementFeatures(
                                data
                            ),
                        };
                    }
                );


            // ==================================================
            // ML Prediction
            // ==================================================

            let records = [];

            let risk = {
                score: null,
                percent: null,
                label: "UNKNOWN",
            };


            try {
                records =
                    await buildRecordsForUser(
                        userId
                    );


                const mlResult =
                    await callMLService(
                        records
                    );


                risk =
                    mapMLToRisk(
                        mlResult
                    );

            } catch (error) {
                console.warn(
                    `Could not compute detailed ML risk for user ${userId}:`,
                    error.response?.data ||
                    error.message
                );
            }


            // ==================================================
            // Response
            // ==================================================

            res
                .status(200)
                .json({
                    id:
                        userId,

                    profile: {
                        displayName:
                            user.displayName ||
                            `${user.firstName || ""} ${user.lastName || ""}`.trim(),

                        firstName:
                            user.firstName ||
                            null,

                        lastName:
                            user.lastName ||
                            null,

                        email:
                            user.email ||
                            null,

                        age:
                            user.age ??
                            null,

                        isPregnant:
                            user.isPregnant ??
                            user.is_pregnant ??
                            null,

                        createdAt:
                            user.createdAt ??
                            user.created_at ??
                            null,

                        homeMetrics:
                            user.homeMetrics ??
                            null,
                    },


                    metrics: {
                        moodSeries,
                        measurementSeries,
                    },


                    mlRecordsUsed:
                        records.length,


                    risk,
                });

        } catch (error) {
            console.error(
                "GET /users/:id error:",
                error
            );


            res
                .status(500)
                .json({
                    error:
                        "Internal server error",
                });
        }
    }
);


// ======================================================
// Export Firebase HTTPS Function
// ======================================================

exports.api =
    functions.https.onRequest(app);