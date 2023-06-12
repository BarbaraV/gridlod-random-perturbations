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
np.random.seed(123)
constr = (2*beta*(beta-alpha)-alpha*(alpha+beta)*np.log(beta/alpha))/(2*np.log(beta/alpha)*alpha*(beta-alpha))
model ={'name': 'check', 'alpha': alpha, 'beta': beta, 'constr': constr}

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

def computeAharm_error(aHarmList, aPert, constr):
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, 0, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def compute_errorHarmT(TInd):
        rPatch = lambda: lod_periodic.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        alphaT = np.zeros(len(aHarmList))
        NFineperEpsilon = world.NWorldFine // Nepsilon
        alphaT[:len(alphaT) - 1] = (rPatch()[np.arange(len(aHarmList) - 1) * np.prod(
            NFineperEpsilon)] - constr*alpha) / (beta - alpha)
        alphaT[len(alphaT) - 1] = constr - np.sum(alphaT[:len(alphaT) - 1])
        #matrix = alpha*np.ones((len(alphaT)-1, len(alphaT)-1)) + (beta-alpha)*np.eye(len(alphaT)-1)
        #alphaT[:len(alphaT) - 1] = np.linalg.solve(matrix,rPatch()[np.arange(len(aHarmList) - 1) * np.prod(NFineperEpsilon)])

        aharmT_combined = np.dot(alphaT,aHarmList)
        aharmT = computeAharm(TInd,aPert)

        return np.abs(aharmT-aharmT_combined)

    error_harmList = list(map(compute_errorHarmT, range(world.NtCoarse)))
    return np.max(error_harmList)

def build_randomunif(Nepsilon,NFine,alpha,beta):
    # builds a random coefficient with spectral bounds alpha and beta, where wiht probability p a uniform value in alpha beta is drawn, otherwise value is alpha
    # piece-wise constant on mesh with Nepsilon blocks
    # returns a fine coefficient on mesh with NFine blocks
    Ntepsilon = np.prod(Nepsilon)
    values = np.random.uniform(alpha, beta, Ntepsilon)

    def randomunif(x):
        index = (x * Nepsilon).astype(int)
        d = np.shape(index)[1]

        if d == 1:
            flatindex = index[:]
        if d == 2:
            flatindex = index[:, 1] * Nepsilon[0] + index[:, 0]
        if d == 3:
            flatindex = index[:, 2] * (Nepsilon[0] * Nepsilon[1]) + index[:, 1] * Nepsilon[0] + index[:, 0]
        else:
            NotImplementedError('other dimensions not available')

        return values[flatindex]

    xFine = util.tCoordinates(NFine)

    return randomunif(xFine).flatten()


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

    harmErrorList = np.zeros(NSamples)
    #absErrorList = np.zeros((len(pList), NSamples))
    relErrorList = np.zeros(NSamples)
    for N in range(NSamples):
        aPert = build_randomunif(Nepsilon,NFine,alpha,beta)

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
        KFullcomb, _, _ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,
                                                               model,compute_indicator=False)
        bFull = basis.T * MFull * f
        uFullcomb, _ = lod_periodic.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        abserror_combined = np.sqrt(
            np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))

        #absErrorList[ii, N] = abserror_combined
        relErrorList[N] = abserror_combined / L2norm
        harmErrorList[N] = computeAharm_error(aharmList, aPert, constr)

    rmsharm = np.sqrt(1. / NSamples * np.sum(harmErrorList ** 2))
    rmserr = np.sqrt(1. / NSamples * np.sum(relErrorList ** 2))
    print("root mean square relative L2-error for solutions over {} samples is: {}".format(NSamples, rmserr))
    print("root mean square error for harmonic mean over {} samples is: {}".format(NSamples, rmsharm))

    sio.savemat('_uniform_meanErr_Nc' + str(Nc) + '.mat',
                {'relErr': relErrorList, 'HarmErr': harmErrorList})
