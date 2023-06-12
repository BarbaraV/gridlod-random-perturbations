import numpy as np
import scipy.io as sio
import time

from gridlod.world import World
from gridlod import util, fem, lod, interp
import algorithms, build_coefficient,lod_periodic

NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([128,128])
NCoarse = np.array([32,32])
k=4
NSamples = 250
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
percentage_comp = 0.15
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList, timeBasis, timeMatrixList, correctorsList\
    = algorithms.computeCSI_offline(world, Nepsilon // NCoarse,k,boundaryConditions,model,correctors=True)
aRef = np.copy(aRefList[-1])
KmsijRef = np.copy(KmsijList[-1])
muTPrimeRef = muTPrimeList[-1]
correctorsRef = correctorsList[-1]

print('offline time for new approach {}'.format(timeBasis+np.sum(np.array(timeMatrixList))))
print('offline time for perturbed LOD {}'.format(timeMatrixList[-1]))

relerr_comb= np.zeros((len(pList), NSamples))
relerr_noup= np.zeros((len(pList), NSamples))
relerr_up = np.zeros((len(pList), NSamples))
relerr_comb_H1Coarse = np.zeros((len(pList), NSamples))
relerr_comb_H1Fine = np.zeros((len(pList), NSamples))
relerr_comb_energyCoarse = np.zeros((len(pList), NSamples))
relerr_comb_energyFine = np.zeros((len(pList), NSamples))
relerr_noup_H1Coarse = np.zeros((len(pList), NSamples))
relerr_noup_H1Fine = np.zeros((len(pList), NSamples))
relerr_noup_energyCoarse = np.zeros((len(pList), NSamples))
relerr_noup_energyFine = np.zeros((len(pList), NSamples))

def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

# LOD for deterministic coeffcient - no updates
basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
MgradFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine)
computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
patchT = list(map(computePatch, range(world.NtCoarse)))
KFullpert = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijRef, periodic=True)
bFull = basis.T * MFull * f
faverage = np.dot(MFull * np.ones(NpFine), f)
uFullpert, _ = lod_periodic.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
uLodCoarsepert = basis * uFullpert
modbasispert = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsRef, periodic=True)
uLodFinepert = modbasispert * uFullpert

ii = 0
for p in pList:
    constr = 1. + p ** 2 * (beta - alpha) / (beta + p * (alpha - beta))
    model['constr'] = constr
    if p == 0.1:
        mean_time_true = 0.
        mean_time_perturbed = 0.
        mean_time_combined = 0.

    for N in range(NSamples):
        aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

        #true LOD
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
        patchRef = lod_periodic.PatchPeriodic(world, k, middle)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        if p == 0.1:
            tic = time.perf_counter()
            patchT, correctorsTtrue, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
            toc = time.perf_counter()
            mean_time_true += (toc-tic)
            correctorsTtrue = tuple(correctorsTtrue)
            modbasistrue = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsTtrue, periodic=True)
        else:
            patchT, correctorsTtrue, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
            correctorsTtrue = tuple(correctorsTtrue)
            modbasistrue = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsTtrue, periodic=True)

        bFull = basis.T * MFull * f
        uFulltrue, _ = lod_periodic.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue
        uLodFinetrue = modbasistrue * uFulltrue

        #combined LOD
        if p == 0.1:
            tic = time.perf_counter()
            _, _, _ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,
                                                                   k,model,compute_indicator=False,
                                                                   correctorsList=None, compute_correc=False)
            toc = time.perf_counter()
            mean_time_combined += (toc-tic)
            KFullcomb, _, correctorsBasiscomb = algorithms.compute_combined_MsStiffness(world, Nepsilon, aPert,
                                                                                        aRefList, KmsijList,
                                                                                        muTPrimeList,
                                                                                        k, model,
                                                                                        compute_indicator=False,
                                                                                        correctorsList=correctorsList,
                                                                                        compute_correc=True)
        else:
            KFullcomb, _, correctorsBasiscomb = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,
                                                                   k,model,compute_indicator=False,
                                                                   correctorsList=correctorsList, compute_correc=True)
        bFull = basis.T * MFull * f
        uFullcomb, _ = lod_periodic.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb
        modbasiscomb = basis - correctorsBasiscomb
        uLodFinecomb = modbasiscomb * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        abs_error_combined = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))
        relerr_comb[ii, N] = abs_error_combined / L2norm
        AFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert)
        H1normFine = np.sqrt(np.dot(uLodFinetrue, MgradFull * uLodFinetrue))
        H1normCoarse = np.sqrt(np.dot(uLodCoarsetrue, MgradFull * uLodCoarsetrue))
        energynormFine = np.sqrt(np.dot(uLodFinetrue, AFull * uLodFinetrue))
        energynormCoarse = np.sqrt(np.dot(uLodCoarsetrue, AFull * uLodCoarsetrue))
        abs_error_combinedH1Fine = np.sqrt(np.dot(uLodFinetrue - uLodFinecomb, MgradFull * (uLodFinetrue - uLodFinecomb)))
        relerr_comb_H1Fine[ii, N] = abs_error_combinedH1Fine / H1normFine
        abs_error_combinedH1Coarse = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecomb, MgradFull * (uLodCoarsetrue - uLodCoarsecomb)))
        relerr_comb_H1Coarse[ii, N] = abs_error_combinedH1Coarse / H1normCoarse
        abs_error_combinedenergyFine = np.sqrt(np.dot(uLodFinetrue - uLodFinecomb, AFull * (uLodFinetrue - uLodFinecomb)))
        relerr_comb_energyFine[ii, N] = abs_error_combinedenergyFine / energynormFine
        abs_error_combinedenergyCoarse = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecomb, AFull * (uLodCoarsetrue - uLodCoarsecomb)))
        relerr_comb_energyCoarse[ii, N] = abs_error_combinedenergyCoarse / energynormCoarse

        # standard LOD no updates
        abs_error_pert = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert)))
        relerr_noup[ii, N] = abs_error_pert / L2norm
        abs_error_pertH1Fine = np.sqrt(np.dot(uLodFinetrue - uLodFinepert, MgradFull * (uLodFinetrue - uLodFinepert)))
        relerr_noup_H1Fine[ii, N] = abs_error_pertH1Fine / H1normFine
        abs_error_pertH1Coarse = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MgradFull * (uLodCoarsetrue - uLodCoarsepert)))
        relerr_noup_H1Coarse[ii, N] = abs_error_pertH1Coarse / H1normCoarse
        abs_error_pertenergyFine = np.sqrt(np.dot(uLodFinetrue - uLodFinepert, AFull * (uLodFinetrue - uLodFinepert)))
        relerr_noup_energyFine[ii, N] = abs_error_pertenergyFine / energynormFine
        abs_error_pertenergyCoarse = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, AFull * (uLodCoarsetrue - uLodCoarsepert)))
        relerr_noup_energyCoarse[ii, N] = abs_error_pertenergyCoarse / energynormCoarse

        # LOD with updates
        if p == 0.1:
            tic = time.perf_counter()
            KFullpertup, _ = algorithms.compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k,
                                                                      percentage_comp)
            toc = time.perf_counter()
            mean_time_perturbed += (toc - tic)
            bFull = basis.T * MFull * f
            uFullpertup, _ = lod_periodic.solvePeriodic(world, KFullpertup, bFull, faverage, boundaryConditions)
            uLodCoarsepertup = basis * uFullpertup
            error_pertup = np.sqrt(
                np.dot(uLodCoarsetrue - uLodCoarsepertup, MFull * (uLodCoarsetrue - uLodCoarsepertup)))
            relerr_up[ii, N] = error_pertup / L2norm

    rmserrNew = np.sqrt(1. / NSamples * np.sum(relerr_comb[ii, :] ** 2))
    rmserrNoup = np.sqrt(1. / NSamples * np.sum(relerr_noup[ii, :] ** 2))
    rmserrNewH1Fine = np.sqrt(1. / NSamples * np.sum(relerr_comb_H1Fine[ii, :] ** 2))
    rmserrNoupH1Fine = np.sqrt(1. / NSamples * np.sum(relerr_noup_H1Fine[ii, :] ** 2))
    rmserrNewH1Coarse = np.sqrt(1. / NSamples * np.sum(relerr_comb_H1Coarse[ii, :] ** 2))
    rmserrNoupH1Coarse = np.sqrt(1. / NSamples * np.sum(relerr_noup_H1Coarse[ii, :] ** 2))
    rmserrNewenergyFine = np.sqrt(1. / NSamples * np.sum(relerr_comb_energyFine[ii, :] ** 2))
    rmserrNoupenergyFine = np.sqrt(1. / NSamples * np.sum(relerr_noup_energyFine[ii, :] ** 2))
    rmserrNewenergyCoarse = np.sqrt(1. / NSamples * np.sum(relerr_comb_energyCoarse[ii, :] ** 2))
    rmserrNoupenergyCoarse = np.sqrt(1. / NSamples * np.sum(relerr_noup_energyCoarse[ii, :] ** 2))
    rmserrUp = np.sqrt(1. / NSamples * np.sum(relerr_up[ii, :] ** 2))
    print("root mean square relative L2-error for new LOD over {} samples for p={} is: {}".format(NSamples,p,rmserrNew))
    print("root mean square relative L2-error for perturbed LOD without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoup))
    print("root mean square relative H1-error for new LODcoarse over {} samples for p={} is: {}".format(NSamples, p, rmserrNewH1Coarse))
    print("root mean square relative H1-error for perturbed LODcoarse without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoupH1Coarse))
    print("root mean square relative H1-error for new LODfine over {} samples for p={} is: {}".format(NSamples,p,rmserrNewH1Fine))
    print("root mean square relative H1-error for perturbed LODfine without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoupH1Fine))
    print("root mean square relative energy error for new LODcoarse over {} samples for p={} is: {}".format(NSamples, p, rmserrNewenergyCoarse))
    print("root mean square relative energy error for perturbed LODcoarse without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoupenergyCoarse))
    print("root mean square relative energy error for new LODfine over {} samples for p={} is: {}".format(NSamples,p,rmserrNewenergyFine))
    print("root mean square relative energy error for perturbed LODfine without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoupenergyFine))
    if p == 0.1:
        print("root mean square relative L2-error for perturbed LOD with {} updates over {} samples for p={} is: {}".
              format(percentage_comp, NSamples, p, rmserrUp))

    ii += 1

    if p == 0.1:
        mean_time_true /= NSamples
        mean_time_perturbed /= NSamples
        mean_time_combined /= NSamples

        print("mean assembly time for standard LOD over {} samples is: {}".format(NSamples, mean_time_true))
        print("mean assembly time for perturbed LOD with {} updates over {} samples is: {}".
              format(NSamples, percentage_comp,mean_time_perturbed))
        print("mean assembly time for new LOD over {} samples is: {}".format(NSamples, mean_time_combined))

sio.savemat('_meanErr2drandcheck.mat', {'relerrNew': relerr_comb,'relerrNoup': relerr_noup, 'relerrUp': relerr_up,
                                        'relerrNewH1Coarse': relerr_comb_H1Coarse,
                                        'relerrNewH1Fine': relerr_comb_H1Fine,
                                        'relerrNoupH1Coarse': relerr_noup_H1Coarse,
                                        'relerrNoupH1Fine': relerr_noup_H1Fine,
                                        'relerrNewenergyCoarse': relerr_comb_energyCoarse,
                                        'relerrNewenergyFine': relerr_comb_energyFine,
                                        'relerrNoupenergyCoarse': relerr_noup_energyCoarse,
                                        'relerrNoupenergyFine': relerr_noup_energyFine,
                                        'pList': pList})