// One-time migration: reassign doctor notes created by the old
// placeholder author ("demo-doctor-001") to a real doctor account.
//
// Usage (from luna-backend/functions):
//   gcloud auth application-default login      # or set GOOGLE_APPLICATION_CREDENTIALS
//   node scripts/migrate-demo-doctor-notes.js <doctor-email>            # dry run
//   node scripts/migrate-demo-doctor-notes.js <doctor-email> --apply    # write changes

const admin = require("firebase-admin");

const PROJECT_ID = "newlunacare";
const LEGACY_AUTHOR_ID = "demo-doctor-001";

admin.initializeApp({ projectId: PROJECT_ID });
const db = admin.firestore();

async function main() {
    const email = process.argv[2];
    const apply = process.argv.includes("--apply");

    if (!email || email.startsWith("--")) {
        console.error(
            "Usage: node scripts/migrate-demo-doctor-notes.js <doctor-email> [--apply]"
        );
        process.exit(1);
    }

    const doctorSnap = await db
        .collection("doctors")
        .where("email", "==", email)
        .limit(1)
        .get();

    if (doctorSnap.empty) {
        console.error(`No doctor found with email ${email}`);
        process.exit(1);
    }

    const doctorDoc = doctorSnap.docs[0];
    const doctor = doctorDoc.data();
    const fullName = [doctor.firstName, doctor.lastName]
        .filter(Boolean)
        .join(" ");
    const authorName = fullName
        ? `Dr. ${fullName}`
        : doctor.email || doctorDoc.id;

    console.log(
        `Target doctor: ${doctorDoc.id} (${authorName})${apply ? "" : "  [DRY RUN]"}`
    );

    const users = await db.collection("users").listDocuments();
    let count = 0;

    for (const userRef of users) {
        const notes = await userRef
            .collection("doctor_notes")
            .where("authorId", "==", LEGACY_AUTHOR_ID)
            .get();

        for (const note of notes.docs) {
            count++;
            console.log(`  users/${userRef.id}/doctor_notes/${note.id}`);

            if (apply) {
                await note.ref.update({
                    authorId: doctorDoc.id,
                    authorName,
                });
            }
        }
    }

    console.log(
        apply
            ? `Updated ${count} note(s).`
            : `${count} note(s) would be updated. Re-run with --apply to write.`
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
