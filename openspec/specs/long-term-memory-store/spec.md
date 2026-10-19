# long-term-memory-store Specification

## Purpose
Stores long-term memory items per user and retrieves the most similar ones by embedding, with
tombstoned deletions so removals can propagate during cross-session sync. Offered both in memory
and as a durable single-file database.

## Requirements

### Requirement: Insert or replace by id
The store SHALL insert a memory item, or replace the stored item with the same id, keeping an independent copy so that later changes to the caller's object do not alter the stored item. Retrieving an item by id MUST return an independent copy, including for tombstoned items, and MUST return nothing for an unknown id.

#### Scenario: Upsert then get
- **WHEN** an item is upserted and then retrieved by its id
- **THEN** the retrieved item has equal fields to the upserted one

#### Scenario: Replace by id
- **WHEN** a second item with the same id but different content is upserted
- **THEN** retrieving the id returns the new content and the store still counts one item

#### Scenario: Defensive copies
- **WHEN** a caller mutates the object it upserted, or an object it retrieved
- **THEN** a subsequent retrieval returns the originally stored values

#### Scenario: Unknown id
- **WHEN** an id that was never stored is retrieved
- **THEN** nothing is returned

### Requirement: Fixed embedding dimension
The store SHALL fix its embedding dimension either at construction or from the first item upserted. Upserting an item or searching with a query whose dimension differs from the fixed dimension MUST raise a ValueError.

#### Scenario: Dimension fixed by first item
- **WHEN** a store without a configured dimension receives a 3-dimensional item and then a 4-dimensional item
- **THEN** the second upsert raises a ValueError

#### Scenario: Dimension fixed at construction
- **WHEN** a store configured with dimension 3 receives a 2-dimensional item
- **THEN** the upsert raises a ValueError

#### Scenario: Query dimension mismatch
- **WHEN** a store holding 3-dimensional items is searched with a 2-dimensional query
- **THEN** a ValueError is raised

### Requirement: Tombstone deletion
Deleting a known item SHALL mark it deleted rather than erase it and SHALL advance its update time to the later of the current clock time and its previous update time plus one microsecond, then report success. Deleting an unknown id MUST report failure and change nothing.

#### Scenario: Delete marks tombstone
- **WHEN** a stored item is deleted at clock time T greater than its update time
- **THEN** delete reports success and retrieving the item shows it deleted with update time T

#### Scenario: Update time strictly increases
- **WHEN** an item whose update time is later than the current clock is deleted
- **THEN** its new update time is its previous update time plus one microsecond

#### Scenario: Delete unknown id
- **WHEN** an id that was never stored is deleted
- **THEN** delete reports failure

### Requirement: Similarity search
Search SHALL return up to k live items belonging to the requested user, optionally restricted to a set of kinds, each paired with its cosine similarity to the query, ordered by similarity descending, then by newer creation time, then by id. A k of zero or less MUST yield an empty result.

#### Scenario: Ranked by similarity
- **WHEN** a user's items are searched with a query closest to item A, then B, then C, with k equal to 2
- **THEN** the result is A then B with their cosine similarities

#### Scenario: User isolation
- **WHEN** two users have items and one user's items are searched
- **THEN** no item of the other user is returned

#### Scenario: Kind filter
- **WHEN** a search is restricted to the fact kind
- **THEN** only items of kind fact are returned

#### Scenario: Deleted items excluded
- **WHEN** an item is deleted and the user is searched
- **THEN** the deleted item is not returned

#### Scenario: Deterministic tie-break
- **WHEN** several items have equal similarity
- **THEN** newer creation times come first and equal creation times are ordered by id

#### Scenario: Non-positive k
- **WHEN** a search is made with k of zero or less
- **THEN** the result is empty

### Requirement: Listing and counting
The store SHALL list all items, optionally filtered by user and including tombstones only on request, and SHALL count live items overall or for one user. Its length MUST equal the number of live items.

#### Scenario: Listing excludes tombstones by default
- **WHEN** a store holds two live items and one deleted item
- **THEN** listing returns the two live items, listing with tombstones returns three, and the length is 2

#### Scenario: Per-user listing and count
- **WHEN** items for two users are stored
- **THEN** listing or counting for one user covers only that user's live items

### Requirement: Durable file-backed store
The file-backed store SHALL persist items, including tombstones, metadata and embeddings, to the given database file so that a store reopened on the same path returns identical items; it MUST also accept an in-memory database and be closable and usable as a context manager.

#### Scenario: Persistence across reopen
- **WHEN** items are upserted and one is deleted, the store is closed, and a new store is opened on the same file
- **THEN** every item, including the tombstone, its metadata and its embedding, is returned unchanged

#### Scenario: Dimension recovered on reopen
- **WHEN** a store reopened on a file holding 3-dimensional items receives a 4-dimensional item
- **THEN** a ValueError is raised

#### Scenario: Context manager closes
- **WHEN** a store is used in a with-block
- **THEN** it is closed on exit and the data remains readable by a new store on the same file
