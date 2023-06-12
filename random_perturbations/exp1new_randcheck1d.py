import numpy as np
import scipy.io as sio

from gridlod.world import World
from gridlod import util, fem, lod, interp
import algorithms, build_coefficient, lod_periodic

NFine = np.array([256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([256])
NList = [8,64]
k=0
NSamples = 500
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeAharm(TInd,a):
    assert(dim==1)
    patch = lod_periodic.PatchPeriodic(world,0,TInd)
    aPatch = lod_periodic.localizeCoefficient(patch, a, periodic=True)
    aPatchHarm = np.sum(1./aPatch)
    return world.NWorldFine/(world.NWorldCoarse * aPatchHarm)

def computeAharm_offline():
    aRefListSingle,_,_,_,_ = algorithms.computeCSI_offline(world, Nepsilon//NCoarse,0,boundaryConditions,model)
    return [world.NWorldFine/(world.NWorldCoarse * np.sum(1./aRefSingle)) for aRefSingle in aRefListSingle]

def computeAharm_error(aHarmList, aPert,constr=1.):
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, 0, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def compute_errorHarmT(TInd):
        rPatch = lambda: lod_periodic.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        alphaT = np.zeros(len(aHarmList))
        NFineperEpsilon = world.NWorldFine // Nepsilon
        alphaT[:len(alphaT) - 1] = (rPatch()[np.arange(len(aHarmList) - 1) * np.prod(
            NFineperEpsilon)] - constr*alpha) / (beta - alpha)
        alphaT[len(alphaT) - 1] = constr - np.sum(alphaT[:len(alphaT) - 1])

        aharmT_combined = np.dot(alphaT,aHarmList)
        aharmT = computeAharm(TInd,aPert)

        return np.abs(aharmT-aharmT_combined)

    error_harmList = list(map(compute_errorHarmT, range(world.NtCoarse)))
    return np.max(error_harmList)


for Nc in NList:
    NCoarse = np.array([Nc])
    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)

    xpFine = util.pCoordinates(NFine)
    ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x)
    f = ffunc(xpFine).flatten()

    aRefList, KmsijList, muTPrimeList, _, _ = algorithms.computeCSI_offline(world, Nepsilon // NCoarse,
                                                                            k, boundaryConditions,model)
    aharmList = computeAharm_offline()

    harmErrorList = np.zeros((len(pList), NSamples))
    absErrorList = np.zeros((len(pList), NSamples))
    relErrorList = np.zeros((len(pList), NSamples))

    ii = 0
    for p in pList:
        constr = 1.+p**2*(beta-alpha)/(beta+p*(alpha-beta))
        model['constr'] = constr
        for N in range(NSamples):
            aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

            MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
            bFull = basis.T * MFull * f
            faverage = np.dot(MFull * np.ones(NpFine), f)

            #true LOD
            middle = NCoarse[0] // 2
            patchRef = lod_periodic.PatchPeriodic(world, k, middle)
            IPatch = lambda: interp.nodalPatchMatrix(patchRef)
            computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

            bFull = basis.T * MFull * f
            uFulltrue, _ = lod_periodic.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
            uLodCoarsetrue = basis * uFulltrue

            #combined LOD
            KFullcomb, _,_ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,
                                                                   model,compute_indicator=False)
            bFull = basis.T * MFull * f
            uFullcomb, _ = lod_periodic.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
            uLodCoarsecomb = basis * uFullcomb

            L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
            abserror_combined = np.sqrt(
                np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))

            absErrorList[ii, N] = abserror_combined
            relErrorList[ii, N] = abserror_combined / L2norm
            harmErrorList[ii, N] = computeAharm_error(aharmList, aPert,constr)

        rmsharm = np.sqrt(1. / NSamples * np.sum(harmErrorList[ii, :] ** 2))
        rmserr = np.sqrt(1. / NSamples * np.sum(relErrorList[ii, :] ** 2))
        print("root mean square relative L2-error for solutions over {} samples for p={} is: {}".format(NSamples, p,
                                                                                                        rmserr))
        print("root mean square error for harmonic mean over {} samples for p={} is: {}".format(NSamples, p, rmsharm))
        ii += 1

    sio.savemat('_meanErr_Nc' + str(Nc) + '.mat',
                {'absErr': absErrorList, 'relErr': relErrorList, 'HarmErr': harmErrorList, 'pList': pList})
