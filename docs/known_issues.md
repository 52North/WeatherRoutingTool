---

## Known Issues

### Speedy IsoBased Routing with Test Data

When using the `speedy_isobased` algorithm with the provided
test weather and depth datasets, routing may terminate early
with the error:

> All pruning segments fully constrained

#### Symptoms
- Routing stops after ~10–11 isochrone steps
- No runtime exception occurs
- No route reaches the destination
- A partial route file is generated with 0% success

#### Cause (Observed)
This occurs when all isochrone pruning segments become constrained
due to the interaction of:
- Pruning parameters
- Environmental constraints (land, depth, map bounds)
- Test data resolution

#### Suggested Workarounds
Users may try:
- Increasing `ISOCHRONE_PRUNE_SEGMENTS`
- Reducing `ISOCHRONE_PRUNE_SECTOR_DEG_HALF`
- Increasing heading resolution
- Reducing pruning strictness
