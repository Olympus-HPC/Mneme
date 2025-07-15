#ifndef PARTICLES_H
#define PARTICLES_H

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

//Simulation parameters
typedef struct{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    unsigned int numCells;
    float3 worldOrigin;
    float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

void integrateSystem(
    float4* d_Pos,
    float4* d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles);

void calcHash(
    unsigned int* d_Hash,
    unsigned int* d_Index,
    float4* d_Pos,
    const simParams_t &params,
    const int numParticles);

void memSet(
    unsigned int* d_Data,
    unsigned int val,
    unsigned int N);

void findCellBoundsAndReorder(
    unsigned int* d_CellStart,
    unsigned int* d_CellEnd,
    float4* d_ReorderedPos,
    float4* d_ReorderedVel,
    unsigned int* d_Hash,
    unsigned int* d_Index,
    float4* d_Pos,
    float4* d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells);

void collide(
    float4* d_Vel,
    float4* d_ReorderedPos,
    float4* d_ReorderedVel,
    unsigned int* d_Index,
    unsigned int* d_CellStart,
    unsigned int* d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells);

void bitonicSort(
    unsigned int* d_DstKey,
    unsigned int* d_DstVal,
    unsigned int* d_SrcKey,
    unsigned int* d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir);
#endif
