import numpy as np
import time

from gridlod import interp, lod
import build_coefficient, lod_periodic, indicator


def computeCSI_offline(world, NepsilonElement, k, boundaryConditions, model, correctors=False):
    dim = np.size(world.NWorldFine)
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
    elif dim == 1:
        middle = world.NWorldCoarse[0] //2
    patch = lod_periodic.PatchPeriodic(world, k, middle)

    tic = time.perf_counter()
    assert(model['name'] in ['check', 'incl', 'inclvalue', 'inclfill', 'inclshift', 'inclLshape'])
    if model['name'] == 'check':
        aRefList = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, NepsilonElement,
                                                             world.NCoarseElement, model['alpha'], model['beta'])
    elif model['name'] == 'incl':
        aRefList = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse,NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'], model['right'])
    elif model['name'] == 'inclvalue':
        aRefList = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model['defval'])
    elif model['name'] in ['inclfill', 'inclshift', 'inclLshape']:
        aRefList = build_coefficient.build_inclusionbasis_change_2d(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model)

    toc = time.perf_counter()
    time_basis = toc-tic


    def computeKmsij(TInd, aPatch, k, boundaryConditions):
        tic = time.perf_counter()
        patch = lod_periodic.PatchPeriodic(world, k, TInd)
        if dim == 1:
            IPatch = lambda: interp.nodalPatchMatrix(patch)
        else:
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        toc = time.perf_counter()
        return patch, correctorsList, csi.Kmsij, csi.muTPrime, toc-tic

    computeSingleKms = lambda aRef: computeKmsij(middle, aRef, k, boundaryConditions)
    if correctors:
        _, correctorsList, KmsijList, muTPrimeList, timeMatrixList = zip(*map(computeSingleKms, aRefList))
        return aRefList, KmsijList, muTPrimeList, time_basis, timeMatrixList, correctorsList
    else:
        _, _, KmsijList, muTPrimeList, timeMatrixList = zip(*map(computeSingleKms, aRefList))
        return aRefList, KmsijList, muTPrimeList, time_basis, timeMatrixList

def compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList, KmsijList,muTPrimeList,k,model,compute_indicator=False,
                                 correctorsList=None, compute_correc=False):
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))
    dim = np.size(world.NWorldFine)

    def compute_combinedT(TInd):
        rPatch = lambda: lod_periodic.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        #direct determination of alpha without optimization
        assert (model['name'] in ['check', 'incl', 'inclvalue', 'inclfill', 'inclshift', 'inclLshape'])

        if model['name'] == 'check':
            alphaT = np.zeros(len(aRefList))
            alpha = model['alpha']
            beta = model['beta']
            try:
               model['constr']
            except:
                constr=1.
            else:
                constr = model['constr']
            NFineperEpsilon = world.NWorldFine//Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse*(Nepsilon//world.NWorldCoarse)
            if dim == 2:
                tmp_indx =np.array([np.arange(len(aRefList)-1)//NEpsilonperPatchCoarse[0],
                                    np.arange(len(aRefList)-1)%NEpsilonperPatchCoarse[0]])
                indx = tmp_indx[0]*NFineperEpsilon[1]*patchT[TInd].NPatchFine[0]+ tmp_indx[1]*NFineperEpsilon[0]
                alphaT[:len(alphaT)-1] = (rPatch()[indx]-constr*alpha)/(beta-alpha)
            elif dim == 1:
                alphaT[:len(alphaT)-1] = (rPatch()[np.arange(len(aRefList)-1)*np.prod(NFineperEpsilon)]-constr*alpha)/(beta-alpha)
            alphaT[len(alphaT)-1] = constr-np.sum(alphaT[:len(alphaT)-1])
        elif model['name'] == 'incl':
            alphaT = np.zeros(len(aRefList))
            bgval = model['bgval']
            inclval = model['inclval']
            blx = model['left'][0]
            bly = model['left'][1]
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0]+blx,
                                 np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0] + bly])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                    + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT)-1] = (inclval - rPatch()[indx])/(inclval-bgval)
            alphaT[len(alphaT)-1] = 1. - np.sum(alphaT[:len(alphaT)-1])
        elif model['name'] == 'inclvalue':
            alphaT = np.zeros(len(aRefList))
            defval = model['defval']
            inclval = model['inclval']
            blx = model['left'][0]
            bly = model['left'][1]
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0] + blx,
                                 np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0] + bly])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                   + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT) - 1] = (inclval - rPatch()[indx]) / (inclval - defval)
            alphaT[len(alphaT) - 1] = 1. - np.sum(alphaT[:len(alphaT) - 1])
        elif model['name'] == 'inclfill':
            alphaT = np.zeros(len(aRefList))
            bgval = model['bgval']
            inclval = model['inclval']
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0],
                                 np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0]])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                    + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT)-1] = (bgval - rPatch()[indx])/(bgval-inclval)
            alphaT[len(alphaT)-1] = 1. - np.sum(alphaT[:len(alphaT)-1])
        elif model['name'] == 'inclshift':
            alphaT = np.zeros(len(aRefList))
            bgval = model['bgval']
            inclval = model['inclval']
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            assert(model['def_bl'][0] < model['left'][0] or model['def_bl'][1]<model['left'][1]
                   or model['def_bl'][0] >= model['right'][0] or model['def_bl'][1]>=model['right'][1])
            # other cases not yet implemented
            blx = model['def_bl'][0]
            bly = model['def_bl'][1]
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0]+blx,
                             np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0] + bly])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT)-1] = (bgval - rPatch()[indx])/(bgval-inclval)
            alphaT[len(alphaT)-1] = 1. - np.sum(alphaT[:len(alphaT)-1])
        elif model['name'] == 'inclLshape':
            alphaT = np.zeros(len(aRefList))
            bgval = model['bgval']
            inclval = model['inclval']
            blx = model['def_bl'][0]
            bly = model['def_bl'][1]
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0]+blx,
                                 np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0] + bly])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                    + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT)-1] = (inclval - rPatch()[indx])/(inclval-bgval)
            alphaT[len(alphaT)-1] = 1. - np.sum(alphaT[:len(alphaT)-1])

        if compute_indicator:
            assert(correctorsList is not None)
            indicatorValue = indicator.computeErrorIndicatorFineMultiple(patchT[TInd],correctorsList,aRefList,alphaT)
            defects = np.sum(alphaT[:len(alphaT) - 1])
            indicatorT = [indicatorValue, defects]
        else:
            indicatorT = None

        if compute_correc:
            assert(correctorsList is not None)
            correctorsT = np.einsum('i, ijk -> jk', alphaT, correctorsList)
        else:
            correctorsT = None

        KmsijT = np.einsum('i, ijk -> jk', alphaT, KmsijList)

        return KmsijT, indicatorT, correctorsT

    KmsijT_list, error_indicator, correctorsT_list = zip(*map(compute_combinedT, range(world.NtCoarse)))

    KmsijT = tuple(KmsijT_list)

    KFull = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)
    if compute_correc:
        correctorsT = tuple(correctorsT_list)
        correcBasis = lod_periodic.assembleBasisCorrectors(world, patchT, correctorsT, periodic=True)
    else:
        correcBasis = None

    return KFull, error_indicator, correcBasis

def compute_perturbed_MsStiffness(world,aPert, aRef, KmsijRef, muTPrimeRef,k, update_percentage):
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))
    dim = np.size(world.NWorldFine)
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    elif dim == 1:
        middle = world.NWorldCoarse[0] // 2
    patchRef = lod_periodic.PatchPeriodic(world, k, middle)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)

    def computeIndicator(TInd):
        aPatch = lambda: lod_periodic.localizeCoefficient(patchT[TInd], aPert, periodic=True)  # true coefficient
        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], muTPrimeRef, aRef, aPatch)

        return E_vh

    def UpdateCorrectors(TInd):
        rPatch = lambda: lod_periodic.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        correctorsList = lod.computeBasisCorrectors(patchT[TInd], IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patchT[TInd], correctorsList, rPatch)

        return patchT[TInd], correctorsList, csi.Kmsij

    def UpdateElements(tol, E, Kmsij_old):
        Elements_to_be_updated = []
        for (i, eps) in E.items():
            if eps > tol:
                Elements_to_be_updated.append(i)

        if np.size(Elements_to_be_updated) != 0:
            patchT_irrelevant, correctorsListT_irrelevant, KmsijTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

            KmsijT_list = list(np.copy(Kmsij_old))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = np.copy(KmsijTNew[i])
                i += 1

            KmsijT = tuple(KmsijT_list)
            return KmsijT
        else:
            return Kmsij_old

    E_vh = list(map(computeIndicator, range(world.NtCoarse)))
    E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0}

    # loop over elements with possible recomputation of correctors
    tol_relative = np.quantile(E_vh, 1.-update_percentage, interpolation='higher')
    KmsijRefList = [KmsijRef for _ in range(world.NtCoarse)] #tile up the stiffness matrix for one element
    KmsijT = UpdateElements(tol_relative, E, KmsijRefList)

    #assembly of matrix
    KFull = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull, E_vh
