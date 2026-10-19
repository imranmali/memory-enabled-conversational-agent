# cross-session-sync Specification

## Purpose
Replicates a user's long-term memories between devices through a shared sync backend so that every
device converges on the same memory set, with deterministic conflict resolution and propagated deletions.

## Requirements

### Requirement: Last-write-wins conflict resolution
When two versions of the same memory (same id) disagree, the system SHALL keep the content of the version with the later `updated_at`, regardless of which side is local and which is remote.

#### Scenario: Newer remote edit wins
- **WHEN** a local memory with `updated_at` 100 is resolved against a remote version with `updated_at` 200
- **THEN** the result carries the remote version's content

#### Scenario: Newer local edit wins
- **WHEN** a local memory with `updated_at` 300 is resolved against a remote version with `updated_at` 200
- **THEN** the result carries the local version's content

#### Scenario: Resolving different memories is rejected
- **WHEN** two memories with different ids are resolved against each other
- **THEN** the system raises a ValueError

### Requirement: Deterministic symmetric tie-break
When both versions have the same `updated_at`, the system MUST pick the version with the lexicographically larger `origin`, and resolution MUST be symmetric: resolving A against B and B against A yields the same content.

#### Scenario: Equal timestamps resolved by origin
- **WHEN** versions from origins "device-a" and "device-b" share the same `updated_at`
- **THEN** the "device-b" version wins whichever side it is passed as

#### Scenario: Fully tied versions resolve identically
- **WHEN** two versions share `updated_at` and `origin` but differ in content
- **THEN** resolving them in either argument order returns identical content

### Requirement: Access statistics merge without counting as a write
The resolved memory SHALL carry the maximum `access_count` and the maximum `last_accessed` of both versions, and these statistics MUST NOT influence which version's content wins.

#### Scenario: Losing side has more accesses
- **WHEN** the older version has `access_count` 7 and `last_accessed` 500 and the newer has `access_count` 2 and `last_accessed` 150
- **THEN** the result has the newer content with `access_count` 7 and `last_accessed` 500

### Requirement: Tombstones are ordinary versions
A deleted memory (`deleted=True`) SHALL take part in conflict resolution like any other version, so a newer deletion beats an older edit and a newer edit beats an older deletion.

#### Scenario: Newer delete beats older edit
- **WHEN** a tombstone with `updated_at` 200 is resolved against an edit with `updated_at` 150
- **THEN** the result is deleted

#### Scenario: Newer edit beats older delete
- **WHEN** an edit with `updated_at` 250 is resolved against a tombstone with `updated_at` 200
- **THEN** the result is live and carries the edit's content

### Requirement: Backend-assigned revisions
A sync backend SHALL keep a per-user revision number that it alone assigns, starting at 0 and increasing monotonically; a push MUST merge each record with last-write-wins and return the latest revision, advancing it only when a stored record changed or a pushed differing version lost (the stored winner is then re-announced so the pusher pulls it).

#### Scenario: Pushing new records advances the revision
- **WHEN** a record is pushed for a user with revision 0
- **THEN** the returned revision is greater than 0

#### Scenario: Pushing unchanged records keeps the revision
- **WHEN** the exact same records are pushed a second time
- **THEN** the returned revision equals the previous one

#### Scenario: Pushing a stale version keeps the newer record
- **WHEN** a version older than the stored record is pushed
- **THEN** the stored record keeps its newer content and is returned by a pull since the previous revision

#### Scenario: Pushing another user's record is rejected
- **WHEN** a push for user "alice" contains a record whose user is "bob"
- **THEN** the backend raises a ValueError and stores nothing from that push

### Requirement: Incremental pull
A backend SHALL return, for a user and a revision, every record changed after that revision together with the latest revision; returned records MUST be copies that cannot alter backend state.

#### Scenario: Pull after a known revision
- **WHEN** records were pushed at revisions 1 and 2 and a client pulls since revision 1
- **THEN** only the record changed at revision 2 is returned along with latest revision 2

#### Scenario: Unknown user pulls nothing
- **WHEN** a user that never pushed pulls since revision 0
- **THEN** the result is an empty list and revision 0

### Requirement: Durable file backend
The file backend SHALL persist each user's records, revision and sessions under a given directory using atomic replace-on-write, so a new backend instance on the same directory sees all previously written data.

#### Scenario: Data survives re-instantiation
- **WHEN** records are pushed through one file backend and a new instance is created on the same directory
- **THEN** pulling from the new instance returns those records and the same revision

### Requirement: Safe storage paths
The file backend MUST derive file names from user ids so that every file lives directly inside the configured directory, whatever characters the user id contains, and distinct user ids MUST map to distinct files.

#### Scenario: Path traversal user id
- **WHEN** a user id such as "../../etc/passwd" pushes records
- **THEN** the only files written are inside the configured directory

#### Scenario: Similar user ids do not collide
- **WHEN** users "a/b" and "a_b" both push records
- **THEN** each pulls back only its own records

### Requirement: Corrupt storage detection
When a user's backend file cannot be parsed or has an invalid structure, the file backend MUST raise a sync storage error naming the file and MUST leave the file unchanged.

#### Scenario: Garbage file contents
- **WHEN** a user's file contains invalid JSON and that user pulls or pushes
- **THEN** a sync storage error is raised and the file content is unchanged

### Requirement: Synchronization converges across devices
A device synchronizer SHALL pull remote changes since its cursor and merge them into its vector store with last-write-wins, then push its new or changed local memories (tombstones included), so that devices that have synced in turn hold identical memories.

#### Scenario: Two devices converge
- **WHEN** devices A and B each create memories and then sync in the order A, B, A
- **THEN** both stores hold the same live memories with identical contents

#### Scenario: Three devices converge
- **WHEN** devices A, B and C each create memories and sync in two rounds
- **THEN** all three stores hold identical memories

### Requirement: Idempotent synchronization
Synchronizing again without any intervening change MUST push nothing, pull nothing and leave the backend revision unchanged.

#### Scenario: Second sync is a no-op
- **WHEN** a device syncs twice in a row with no changes anywhere
- **THEN** the second report shows 0 pushed, 0 pulled, 0 conflicts and the same revision

### Requirement: Deletions propagate
A memory deleted on one device SHALL become deleted on every other device after both have synced, and MUST NOT be resurrected by a stale copy.

#### Scenario: Delete on one device
- **WHEN** device A deletes a memory that device B also holds and A then B sync
- **THEN** the memory is deleted in B's store and absent from B's live memories

### Requirement: Concurrent edits converge to the winner
When two devices change the same memory before syncing, every device SHALL end with the last-write-wins winner, and the device that discovers the clash MUST count it as a conflict in its report.

#### Scenario: Concurrent edit of one memory
- **WHEN** devices A and B edit the same memory at times 200 and 300 and sync in the order A, B, A
- **THEN** both stores hold B's content and B's report counts one conflict

#### Scenario: Losing push is corrected
- **WHEN** a device pushes an edit whose `updated_at` is older than the version already in the backend
- **THEN** the same sync replaces the device's copy with the backend's winner and later syncs push nothing

### Requirement: User isolation
Synchronizing one user MUST neither read nor write memories belonging to any other user.

#### Scenario: Two users on one backend
- **WHEN** a device syncs user "alice" while its store also holds memories of user "bob"
- **THEN** no memory of "bob" is pushed and another device syncing "bob" receives nothing of "alice"

### Requirement: Origin stamping
Before pushing, the synchronizer SHALL replace the placeholder origin "local" with its device id in both the pushed record and its own store; the device id MUST be non-empty and MUST NOT be "local".

#### Scenario: Local memory gets the device id
- **WHEN** device "phone" pushes a memory created with origin "local"
- **THEN** the backend and the phone's store both record origin "phone"

#### Scenario: Invalid device id
- **WHEN** a synchronizer is created with device id "local" or an empty string
- **THEN** a ValueError is raised

### Requirement: Persistable sync state
The synchronizer SHALL expose its per-user cursor state as a JSON-serialisable mapping and accept it back, so a restarted synchronizer resumes without re-pushing or re-pulling; state from another device MUST be rejected.

#### Scenario: Restored synchronizer resumes
- **WHEN** a synchronizer's state is saved, loaded into a new instance and that instance syncs without changes
- **THEN** it pushes 0 and pulls 0

#### Scenario: State from another device
- **WHEN** state saved by device "a" is loaded into the synchronizer of device "b"
- **THEN** a ValueError is raised

### Requirement: Recovery from backend reset
If the backend reports a revision lower than the device's cursor, the synchronizer SHALL treat the backend as reset and re-push all of the user's local memories.

#### Scenario: Backend replaced by an empty one
- **WHEN** a synced device is pointed at a fresh empty backend and syncs
- **THEN** all its memories for the user are pushed to the new backend
