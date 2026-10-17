/**
 * Firebase Functions backend for LunaCare Doctor Portal
 */

const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { FieldValue } = require("firebase-admin/firestore");
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
const DOCTORS_COLLECTION = "doctors";

/**
 * Cloud Run XGBoost prediction service in the NewLunaCare project.
 *
 * The Cloud Run service reads ml_models/current from Firestore
 * and dynamically loads the latest published .pkl model from
 * Firebase Storage / Google Cloud Storage.
 */
const ML_URL =
    process.env.ML_URL ||
    "https://luna-ml-service-47333217295.us-central1.run.app/predict";


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

async function callMLService(records) {
    const response = await axios.post(ML_URL, {
        records,
    });

    return response.data;
}


function mapMLToRisk(mlResult) {
    const score = Number(
        mlResult?.risk_score
    );

    const modelVersion =
        mlResult?.model_version ?? null;

    if (!Number.isFinite(score)) {
        return {
            score: null,
            percent: null,
            label: "UNKNOWN",
            modelVersion,
        };
    }

    return {
        score,
        percent: Math.round(score * 100),
        label: "PPD RISK SCORE",
        modelVersion,
    };
}


// ======================================================
// Doctor Authentication / Authorization
// ======================================================

/**
 * Frontend must send:
 *
 * Authorization: Bearer <firebase-id-token>
 */
async function authenticateDoctor(req, res, next) {
    const authHeader =
        req.headers.authorization || "";

    if (!authHeader.startsWith("Bearer ")) {
        return res.status(401).json({
            error: "Authentication required",
        });
    }

    const idToken =
        authHeader.substring(7);

    try {
        const decodedToken =
            await admin
                .auth()
                .verifyIdToken(idToken);

        const doctorSnap =
            await db
                .collection(
                    DOCTORS_COLLECTION
                )
                .where(
                    "authUid",
                    "==",
                    decodedToken.uid
                )
                .limit(1)
                .get();

        if (doctorSnap.empty) {
            return res.status(403).json({
                error:
                    "Doctor account not found",
            });
        }

        const doctorDoc =
            doctorSnap.docs[0];

        const doctor =
            doctorDoc.data();

        if (doctor.role !== "doctor") {
            return res.status(403).json({
                error:
                    "Doctor access required",
            });
        }

        if (doctor.authorized !== true) {
            return res.status(403).json({
                error:
                    "Doctor is not authorized",
            });
        }

        req.doctor = {
            id: doctorDoc.id,
            authUid:
                decodedToken.uid,
            ...doctor,
        };

        next();

    } catch (error) {
        console.error(
            "Doctor authentication error:",
            error
        );

        return res.status(401).json({
            error:
                "Invalid authentication token",
        });
    }
}


/**
 * Check whether this doctor is assigned
 * to the requested patient.
 */
function doctorCanAccessPatient(
    doctor,
    patientId
) {
    return (
        Array.isArray(
            doctor?.patientIds
        ) &&
        doctor.patientIds.includes(
            patientId
        )
    );
}


function requireAssignedPatient(
    req,
    res,
    next
) {
    const patientId =
        req.params.id;

    if (
        !doctorCanAccessPatient(
            req.doctor,
            patientId
        )
    ) {
        return res.status(403).json({
            error:
                "You do not have access to this patient",
        });
    }

    next();
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


function pickNumber(object, paths) {
    for (const path of paths) {
        const value = Array.isArray(path)
            ? readPath(object, path)
            : object?.[path];

        const number =
            numberOrNull(value);

        if (number !== null) {
            return number;
        }
    }

    return null;
}


function getDateKey(value) {
    if (!value) {
        return null;
    }

    let date;

    if (
        typeof value.toDate === "function"
    ) {
        date = value.toDate();
    } else if (
        value instanceof Date
    ) {
        date = value;
    } else {
        date = new Date(value);
    }

    if (
        Number.isNaN(date.getTime())
    ) {
        return null;
    }

    return date
        .toISOString()
        .slice(0, 10);
}


function getTimeValue(value) {
    if (!value) {
        return 0;
    }

    if (
        typeof value.toDate === "function"
    ) {
        return value
            .toDate()
            .getTime();
    }

    if (
        value instanceof Date
    ) {
        return value.getTime();
    }

    const date =
        new Date(value);

    if (
        Number.isNaN(date.getTime())
    ) {
        return 0;
    }

    return date.getTime();
}


function deepMerge(target, source) {
    const result = {
        ...(target || {}),
    };

    for (
        const [key, value]
        of Object.entries(source || {})
    ) {
        if (
            value &&
            typeof value === "object" &&
            !Array.isArray(value) &&
            typeof value.toDate !== "function"
        ) {
            result[key] =
                deepMerge(
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

function mapMoodTo1to5(rawMood) {
    const mood =
        Number(rawMood);

    if (
        !Number.isFinite(mood)
    ) {
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
        .orderBy(
            dateField,
            "desc"
        )
        .limit(1)
        .get();

    if (snap.empty) {
        return null;
    }

    const doc =
        snap.docs[0];

    return {
        id: doc.id,
        ...doc.data(),
    };
}


async function getLastActive(userRef) {
    const [
        moodLog,
        measurementLog,
    ] = await Promise.all([
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

    if (
        moodLog?.createdAt
    ) {
        dates.push(
            getTimeValue(
                moodLog.createdAt
            )
        );
    }

    if (
        measurementLog?.createdAt
    ) {
        dates.push(
            getTimeValue(
                measurementLog.createdAt
            )
        );
    }

    const validDates =
        dates.filter(
            (value) => value > 0
        );

    if (
        validDates.length === 0
    ) {
        return null;
    }

    return new Date(
        Math.max(...validDates)
    );
}


async function getMoodTrend(
    userRef,
    limit = 10
) {
    const snap = await userRef
        .collection("mood_logs")
        .orderBy(
            "createdAt",
            "desc"
        )
        .limit(limit)
        .get();

    if (snap.empty) {
        return [];
    }

    return snap.docs
        .map(
            (doc) => doc.data()
        )
        .reverse()
        .map(
            (data) =>
                typeof data.mood ===
                    "number"
                    ? data.mood
                    : null
        )
        .filter(
            (value) =>
                value !== null
        );
}


// ======================================================
// Measurement Mapping
// ======================================================

function extractMeasurementFeatures(
    data
) {
    return {
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

async function buildRecordsForUser(
    userId
) {
    const userRef = db
        .collection(
            USERS_COLLECTION
        )
        .doc(userId);

    const [
        userDoc,
        moodSnap,
        symptomSnap,
        measurementSnap,
    ] = await Promise.all([
        userRef.get(),

        userRef
            .collection(
                "mood_logs"
            )
            .orderBy(
                "createdAt",
                "desc"
            )
            .limit(30)
            .get(),

        userRef
            .collection(
                "symptom_logs"
            )
            .orderBy(
                "createdAt",
                "desc"
            )
            .limit(30)
            .get(),

        userRef
            .collection(
                "measurements"
            )
            .orderBy(
                "createdAt",
                "desc"
            )
            .limit(30)
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


    const moodByDate =
        new Map();

    const moodDocs =
        [...moodSnap.docs]
            .reverse();


    for (
        const doc of moodDocs
    ) {
        const data =
            doc.data();

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


    const symptomByDate =
        new Map();

    const symptomDocs =
        [...symptomSnap.docs]
            .reverse();


    for (
        const doc of symptomDocs
    ) {
        const data =
            doc.data();

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


    const measurementsByDate =
        new Map();

    const measurementDocs =
        [...measurementSnap.docs]
            .reverse();


    for (
        const doc
        of measurementDocs
    ) {
        const data =
            doc.data();

        const dateKey =
            getDateKey(
                data.createdAt ??
                data.date
            );

        if (!dateKey) {
            continue;
        }

        const current =
            measurementsByDate
                .get(dateKey) || {};

        measurementsByDate.set(
            dateKey,
            deepMerge(
                current,
                data
            )
        );
    }


    const allDates = [
        ...measurementsByDate.keys(),
    ]
        .sort()
        .slice(-30);


    if (
        allDates.length === 0
    ) {
        throw new Error(
            `No usable dated health records found for ${userId}`
        );
    }


    const records = [];


    for (
        const date
        of allDates
    ) {
        const moodData =
            moodByDate.get(date) ||
            {};

        const symptomData =
            symptomByDate.get(date) ||
            {};

        const symptomValues =
            symptomData.values ||
            {};

        const measurementData =
            measurementsByDate
                .get(date) ||
            {};

        const measurementFeatures =
            extractMeasurementFeatures(
                measurementData
            );


        const record = {
            ...measurementFeatures,

            mood_1to5:
                measurementFeatures
                    .mood_1to5 ??
                mapMoodTo1to5(
                    moodData.mood
                ),

            fatigue_1to10:
                numberOrNull(
                    symptomValues[
                    "Fatigue"
                    ]
                ) ??
                measurementFeatures
                    .fatigue_1to10,

            appetite_issue_1to10:
                measurementFeatures
                    .appetite_issue_1to10 ??
                numberOrNull(
                    symptomValues[
                    "Appetite"
                    ]
                ),

            bleeding_1to10:
                measurementFeatures
                    .bleeding_1to10 ??
                numberOrNull(
                    symptomValues[
                    "Bleeding"
                    ]
                ),

            hair_loss_1to10:
                measurementFeatures
                    .hair_loss_1to10 ??
                numberOrNull(
                    symptomValues[
                    "Hair Loss"
                    ]
                ),

            sleep_trouble_1to10:
                measurementFeatures
                    .sleep_trouble_1to10 ??
                numberOrNull(
                    symptomValues[
                    "Sleep Trouble"
                    ]
                ),
        };


        const completeRecord = {};

        for (
            const field
            of MODEL_FIELDS
        ) {
            completeRecord[field] =
                record[field] ??
                null;
        }

        records.push(
            completeRecord
        );
    }


    return records;
}


// ======================================================
// Doctor Profile Route
// ======================================================

/**
 * GET /me
 *
 * Returns the currently logged-in doctor.
 */
app.get(
    "/me",

    authenticateDoctor,

    async (req, res) => {
        const doctor =
            req.doctor;

        res.status(200).json({
            doctor: {
                id:
                    doctor.id,

                firstName:
                    doctor.firstName ??
                    null,

                lastName:
                    doctor.lastName ??
                    null,

                email:
                    doctor.email ??
                    null,

                phone:
                    doctor.phone ??
                    null,

                organization:
                    doctor.organization ??
                    null,

                specialty:
                    doctor.specialty ??
                    null,

                cpsoNumber:
                    doctor.cpsoNumber ??
                    null,

                accessLevel:
                    doctor.accessLevel ??
                    null,

                role:
                    doctor.role ??
                    null,

                authorized:
                    doctor.authorized === true,

                patientCount:
                    Array.isArray(
                        doctor.patientIds
                    )
                        ? doctor.patientIds.length
                        : 0,
            },
        });
    }
);


// ======================================================
// Dashboard Route
// ======================================================

/**
 * GET /users
 *
 * Returns ONLY patients assigned to the logged-in doctor.
 */
app.get(
    "/users",

    authenticateDoctor,

    async (req, res) => {
        try {
            const patientIds =
                Array.isArray(
                    req.doctor.patientIds
                )
                    ? req.doctor.patientIds
                    : [];


            if (
                patientIds.length === 0
            ) {
                return res
                    .status(200)
                    .json({
                        users: [],
                    });
            }


            const patientDocs =
                await Promise.all(
                    patientIds.map(
                        (patientId) =>
                            db
                                .collection(
                                    USERS_COLLECTION
                                )
                                .doc(
                                    patientId
                                )
                                .get()
                    )
                );


            const users = [];


            for (
                const doc
                of patientDocs
            ) {
                if (!doc.exists) {
                    console.warn(
                        `Assigned patient ${doc.id} does not exist`
                    );

                    continue;
                }


                const data =
                    doc.data();

                const userId =
                    doc.id;

                const userRef =
                    doc.ref;


                let risk = {
                    score: null,
                    percent: null,
                    label: "UNKNOWN",
                    modelVersion: null,
                };


                let lastActive =
                    null;

                let moodTrend =
                    [];


                try {
                    const [
                        records,
                        active,
                        trend,
                    ] =
                        await Promise.all([
                            buildRecordsForUser(
                                userId
                            ),

                            getLastActive(
                                userRef
                            ),

                            getMoodTrend(
                                userRef,
                                10
                            ),
                        ]);


                    lastActive =
                        active;

                    moodTrend =
                        trend;


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
                        data.email ||
                        null,

                    age:
                        data.age ??
                        null,

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

                    modelVersion:
                        risk.modelVersion,

                    moodTrend,
                });
            }


            res
                .status(200)
                .json({
                    users,
                });

        } catch (error) {
            console.error(
                "GET /users error:",
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
// Patient Detail Route
// ======================================================

/**
 * GET /users/:id
 */
app.get(
    "/users/:id",

    authenticateDoctor,
    requireAssignedPatient,

    async (req, res) => {
        const userId =
            req.params.id;


        try {
            const userRef =
                db
                    .collection(
                        USERS_COLLECTION
                    )
                    .doc(
                        userId
                    );


            const userDoc =
                await userRef.get();


            if (
                !userDoc.exists
            ) {
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
            ] =
                await Promise.all([
                    userRef
                        .collection(
                            "mood_logs"
                        )
                        .orderBy(
                            "createdAt",
                            "asc"
                        )
                        .get(),

                    userRef
                        .collection(
                            "measurements"
                        )
                        .orderBy(
                            "createdAt",
                            "asc"
                        )
                        .get(),
                ]);


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


            let records = [];

            let risk = {
                score: null,
                percent: null,
                label: "UNKNOWN",
                modelVersion: null,
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
// Doctor Notes
// ======================================================


// ======================================================
// CREATE Doctor Note
// POST /users/:id/doctor-notes
// ======================================================

app.post(
    "/users/:id/doctor-notes",

    authenticateDoctor,
    requireAssignedPatient,

    async (req, res) => {
        const userId =
            req.params.id;

        const text =
            String(
                req.body?.text ||
                ""
            ).trim();


        if (!text) {
            return res
                .status(400)
                .json({
                    error:
                        "Note text is required",
                });
        }


        if (
            text.length > 1000
        ) {
            return res
                .status(400)
                .json({
                    error:
                        "Note cannot exceed 1000 characters",
                });
        }


        try {
            const userRef =
                db
                    .collection(
                        USERS_COLLECTION
                    )
                    .doc(
                        userId
                    );


            const userDoc =
                await userRef.get();


            if (
                !userDoc.exists
            ) {
                return res
                    .status(404)
                    .json({
                        error:
                            "User not found",
                    });
            }


            const noteRef =
                userRef
                    .collection(
                        "doctor_notes"
                    )
                    .doc();


            const doctorFullName =
                [
                    req.doctor.firstName,
                    req.doctor.lastName,
                ]
                    .filter(Boolean)
                    .join(" ");

            const authorName =
                doctorFullName
                    ? `Dr. ${doctorFullName}`
                    : req.doctor.email ||
                    req.doctor.id;


            const note = {
                text,

                authorId:
                    req.doctor.id,

                authorName,

                createdAt:
                    FieldValue
                        .serverTimestamp(),

                updatedAt:
                    FieldValue
                        .serverTimestamp(),
            };


            await noteRef.set(
                note
            );


            const savedNote =
                await noteRef.get();


            res
                .status(201)
                .json({
                    id:
                        savedNote.id,

                    ...savedNote.data(),
                });

        } catch (error) {
            console.error(
                "POST doctor note error:",
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
// READ Doctor Notes
// GET /users/:id/doctor-notes
// ======================================================

app.get(
    "/users/:id/doctor-notes",

    authenticateDoctor,
    requireAssignedPatient,

    async (req, res) => {
        const userId =
            req.params.id;


        try {
            const userRef =
                db
                    .collection(
                        USERS_COLLECTION
                    )
                    .doc(
                        userId
                    );


            const userDoc =
                await userRef.get();


            if (
                !userDoc.exists
            ) {
                return res
                    .status(404)
                    .json({
                        error:
                            "User not found",
                    });
            }


            const snap =
                await userRef
                    .collection(
                        "doctor_notes"
                    )
                    .orderBy(
                        "createdAt",
                        "desc"
                    )
                    .get();


            const notes =
                snap.docs.map(
                    (doc) => ({
                        id:
                            doc.id,

                        ...doc.data(),
                    })
                );


            res
                .status(200)
                .json({
                    notes,
                });

        } catch (error) {
            console.error(
                "GET doctor notes error:",
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
// UPDATE Doctor Note
// PATCH /users/:id/doctor-notes/:noteId
// ======================================================

app.patch(
    "/users/:id/doctor-notes/:noteId",

    authenticateDoctor,
    requireAssignedPatient,

    async (req, res) => {
        const userId =
            req.params.id;

        const noteId =
            req.params.noteId;

        const text =
            String(
                req.body?.text ||
                ""
            ).trim();


        if (!text) {
            return res
                .status(400)
                .json({
                    error:
                        "Note text is required",
                });
        }


        if (
            text.length > 1000
        ) {
            return res
                .status(400)
                .json({
                    error:
                        "Note cannot exceed 1000 characters",
                });
        }


        try {
            const noteRef =
                db
                    .collection(
                        USERS_COLLECTION
                    )
                    .doc(
                        userId
                    )
                    .collection(
                        "doctor_notes"
                    )
                    .doc(
                        noteId
                    );


            const noteDoc =
                await noteRef.get();


            if (
                !noteDoc.exists
            ) {
                return res
                    .status(404)
                    .json({
                        error:
                            "Note not found",
                    });
            }


            const noteData =
                noteDoc.data();


            if (
                noteData.authorId !==
                req.doctor.id
            ) {
                return res
                    .status(403)
                    .json({
                        error:
                            "You cannot edit another doctor's note",
                    });
            }


            await noteRef.update({
                text,

                updatedAt:
                    FieldValue
                        .serverTimestamp(),
            });


            const updatedNote =
                await noteRef.get();


            res
                .status(200)
                .json({
                    id:
                        updatedNote.id,

                    ...updatedNote.data(),
                });

        } catch (error) {
            console.error(
                "PATCH doctor note error:",
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
// DELETE Doctor Note
// DELETE /users/:id/doctor-notes/:noteId
// ======================================================

app.delete(
    "/users/:id/doctor-notes/:noteId",

    authenticateDoctor,
    requireAssignedPatient,

    async (req, res) => {
        const userId =
            req.params.id;

        const noteId =
            req.params.noteId;


        try {
            const noteRef =
                db
                    .collection(
                        USERS_COLLECTION
                    )
                    .doc(
                        userId
                    )
                    .collection(
                        "doctor_notes"
                    )
                    .doc(
                        noteId
                    );


            const noteDoc =
                await noteRef.get();


            if (
                !noteDoc.exists
            ) {
                return res
                    .status(404)
                    .json({
                        error:
                            "Note not found",
                    });
            }


            const noteData =
                noteDoc.data();


            if (
                noteData.authorId !==
                req.doctor.id
            ) {
                return res
                    .status(403)
                    .json({
                        error:
                            "You cannot delete another doctor's note",
                    });
            }


            await noteRef.delete();


            res
                .status(200)
                .json({
                    message:
                        "Note deleted successfully",
                });

        } catch (error) {
            console.error(
                "DELETE doctor note error:",
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