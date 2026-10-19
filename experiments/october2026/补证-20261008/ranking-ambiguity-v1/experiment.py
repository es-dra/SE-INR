"""Frozen synthetic ranking study. No import-time experiments or GPU use."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import signal
import time
import warnings
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.stats import chi2

warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
HERE = Path(__file__).resolve().parent
MULT = np.array([2., 2., 2., 1.])
LP_CALLS = 0
LP_SECONDS = 0.
DEADLINE = float('inf')
STATE = {}
CERT_TOL = 1e-6


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def journal(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')


def interrupted(signum, frame):
    raise TimeoutError(f'Run interrupted by signal {signum}')


def checkpoint():
    if time.time() >= DEADLINE:
        raise TimeoutError('Predeclared absolute CPU-stage deadline reached')


def components():
    t = np.arange(8)
    blocks = []
    for k in range(1, 4):
        F = np.stack([np.cos(2*np.pi*k*t/8), np.sin(2*np.pi*k*t/8)], axis=1)/2
        blocks.append(F @ F.T)
    f = (-1.)**t / np.sqrt(8)
    blocks.append(np.outer(f, f))
    assert np.max(np.abs(np.array([np.trace(c) for c in blocks])-MULT)) < 1e-12
    return np.array(blocks)


Q = components()


def shape(name):
    p = np.arange(1., 5.)**(-2 if name != 'power3' else -3)
    if name == 'power2_swap_k1_k3':
        p[[0, 2]] = p[[2, 0]]
    if name == 'middle_spike':
        p = np.array([.1, 3., .1, .8])
    return p * 8/(MULT @ p)


def matrices(phases, nu, arms):
    A = np.eye(8)[[site for phase in phases for site in [phase, phase+4]]]
    O = np.einsum('ai,kij,bj->kab', A, Q, A)
    B = []
    for arm in arms:
        C = np.einsum('k,kij->ij', shape(arm), Q)
        S = A @ C @ A.T + nu*np.eye(len(A))
        B.append(np.linalg.solve(S, A @ C).T)
    return A, O, B


def affine(B, A, nu, W):
    E = W @ (B @ A - np.eye(8))
    return np.array([np.trace(E @ c @ E.T) for c in Q]), float(nu*np.sum((W@B)**2))


def endpoints(coeff, M, lower, upper):
    global LP_CALLS, LP_SECONDS
    ends, witnesses, residuals, dualities = [], [], [], []
    Au, bu = np.vstack([M, -M]), np.r_[upper, -lower]
    for sign in [1., -1.]:
        checkpoint()
        LP_CALLS += 1
        lp_started = time.monotonic()
        fit = linprog(sign*coeff, A_ub=Au, b_ub=bu,
                      A_eq=MULT[None], b_eq=[8.], bounds=[(0, 8)]*4,
                      method='highs', options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                                              'dual_feasibility_tolerance': 1e-9})
        LP_SECONDS += time.monotonic()-lp_started
        if not fit.success:
            return {'status': 'empty' if fit.status == 2 else 'solver_failure', 'message': fit.message}
        p = fit.x
        residual = max(float(np.max(M@p-upper)), float(np.max(lower-M@p)),
                       abs(float(MULT@p)-8), float(np.max(-p)), 0.)
        if residual > 1e-7:
            return {'status': 'witness_failure', 'residual': residual}
        # Weak-duality lower bound with sign-clipped multipliers. Bound
        # stationarity error over 0<=p<=8, then expand for floating arithmetic.
        lam = np.minimum(fit.ineqlin.marginals, 0.)
        eta = np.minimum(fit.upper.marginals, 0.)
        mu = np.maximum(fit.lower.marginals, 0.)
        zeta = float(fit.eqlin.marginals[0])
        remainder = sign*coeff-Au.T@lam-MULT*zeta-eta-mu
        primal = float(sign*coeff@p)
        dual = float(bu@lam+8*zeta+8*np.sum(eta)+8*np.minimum(remainder,0).sum())
        pad = 1e-8*max(1.,abs(primal),abs(dual))
        outer = dual-pad
        if outer > primal+1e-10 or primal-outer > 1e-6:
            return {'status':'dual_bound_failure','primal':primal,'expanded_dual':outer}
        ends.append(outer if sign == 1 else -outer)
        witnesses.append(p.tolist())
        residuals.append(residual)
        dualities.append({'primal':primal,'dual_with_box_stationarity_correction':dual,
                         'outward_pad':pad,'expanded_duality_gap':primal-outer})
    return {'status': 'ok', 'lower': ends[0], 'upper': ends[1],
            'witnesses': witnesses, 'max_residual': max(residuals), 'dualities':dualities}


def gaps(coeffs, M, lower, upper):
    output, U = [], np.zeros((len(coeffs), len(coeffs)))
    for j, k in itertools.combinations(range(len(coeffs)), 2):
        c = coeffs[j][0]-coeffs[k][0]
        offset = coeffs[j][1]-coeffs[k][1]
        e = endpoints(c, M, lower, upper)
        if e['status'] != 'ok':
            return e, None
        e.update(j=j, k=k, lower=e['lower']+offset, upper=e['upper']+offset)
        output.append(e)
        U[j, k], U[k, j] = e['upper'], -e['lower']
    return {'status': 'ok', 'pairs': output}, U


def directions(m):
    D = list(np.eye(m))
    for i, j in itertools.combinations(range(m), 2):
        D += [np.eye(m)[i]+np.eye(m)[j], np.eye(m)[i]-np.eye(m)[j]]
    return np.array(D)


def run(root):
    global DEADLINE
    protocol_bytes = (HERE/'protocol.json').read_bytes()
    P = json.loads(protocol_bytes)
    root.mkdir(parents=True, exist_ok=False)
    began = time.time()
    STATE.update(stage='prepare', started_unix=began, selected=0, scored=0, exact_groups=0)
    DEADLINE = began+P['resources']['absolute_wall_seconds']
    frozen = {'started_unix': began, 'deadline_unix': DEADLINE,
              'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cohort_size': P['cohort_size']}
    save(root/'frozen.json', frozen)
    (root/'protocol.json').write_bytes(protocol_bytes)
    print('FROZEN', json.dumps(frozen), flush=True)
    ledger, truths, observations = [], {}, {}
    rng = np.random.default_rng(P['seed'])
    combinations = list(itertools.product(P['phase_sets'], P['noise_variance'], P['truth_spectra'],
                                          P['observational_replicates_N'], range(P['mc_repeats'])))
    rng.shuffle(combinations)
    for idx, (phases, nu, name, N, repeat) in enumerate(combinations):
        checkpoint()
        cid = f'case{idx:04d}'
        A, O, _ = matrices(phases, nu, P['candidate_spectra'])
        p = shape(name)
        S = np.einsum('k,kij->ij', p, O)+nu*np.eye(len(A))
        observations[cid] = rng.standard_normal((N, len(A))) @ np.linalg.cholesky(S).T
        ledger.append({'id':cid, 'phases':phases, 'noise_variance':nu, 'N':N, 'repeat':repeat})
        truths[cid] = {'p':p.tolist(), 'name':name}
    assert len(ledger) == P['cohort_size']
    np.savez_compressed(root/'observations.npz', **observations)
    save(root/'case_metadata.json', ledger)
    save(root/'evaluation_truth.json', truths)
    del observations, truths
    # Selection opens only observed samples and opaque metadata.
    all_y = np.load(root/'observations.npz')
    selections = []
    STATE['stage'] = 'select'
    for idx, meta in enumerate(ledger):
        checkpoint()
        started = time.monotonic()
        A, O, B = matrices(meta['phases'], meta['noise_variance'], P['candidate_spectra'])
        y, N, nu = all_y[meta['id']], meta['N'], meta['noise_variance']
        D = directions(len(A))
        M = np.einsum('ai,kij,aj->ak', D, O, D)
        vhat = np.mean((y @ D.T)**2, axis=0)
        alpha = .05/len(D)
        offset = nu*np.sum(D**2, axis=1)
        lo = N*vhat/chi2.ppf(1-alpha/2, N)-offset
        hi = N*vhat/chi2.ppf(alpha/2, N)-offset
        coeffs = [affine(b, A, nu, np.eye(8)) for b in B]
        e, U = gaps(coeffs, M, lo, hi)
        certified = [] if U is None else [j for j in range(len(B)) if np.max(U[j]) <= CERT_TOL]
        # Gaussian evidence is a classic baseline, not a risk guarantee.
        values = []
        for arm in P['candidate_spectra']:
            S = np.einsum('k,kij->ij', shape(arm), O)+nu*np.eye(len(A))
            values.append(float(np.linalg.slogdet(S)[1]+np.trace(np.linalg.solve(S, y.T@y/N))))
        selections.append({**meta, 'intervals':e, 'certified_arm':certified[0] if certified else None,
                           'eb_arm':int(np.argmin(values)), 'eb_scores':values,
                           'direction_lower':lo.tolist(), 'direction_upper':hi.tolist(),
                           'selection_seconds':time.monotonic()-started})
        journal(root/'selection_journal.jsonl', selections[-1])
        STATE['selected'] = len(selections)
        if (idx+1) % 100 == 0:
            save(root/'selection_partial.json', selections)
            print('SELECT', idx+1, '/', len(ledger), 'LP', LP_CALLS, flush=True)
    all_y.close()
    save(root/'selections.json', selections)
    digest = hashlib.sha256((root/'selections.json').read_bytes()).hexdigest()
    save(root/'selection_cohort.json', {'cohort_size':len(selections), 'selections_sha256':digest,
                                      'frozen_before_truth_score_unix':time.time()})
    print('COHORT_FROZEN', digest, flush=True)
    STATE.update(stage='score', selection_lp_calls=LP_CALLS, selection_lp_seconds=LP_SECONDS)
    truths = json.loads((root/'evaluation_truth.json').read_text())
    scores, exact = [], {}
    for selected in selections:
        checkpoint()
        A, O, B = matrices(selected['phases'], selected['noise_variance'], P['candidate_spectra'])
        p = np.array(truths[selected['id']]['p'])
        D = directions(len(A))
        M = np.einsum('ai,kij,aj->ak', D, O, D)
        covered = bool(np.all(M@p >= np.array(selected['direction_lower'])-1e-7)
                       and np.all(M@p <= np.array(selected['direction_upper'])+1e-7))
        risk = np.array([a@p+b for a,b in [affine(b,A,selected['noise_variance'],np.eye(8)) for b in B]])
        j = selected['certified_arm']
        scores.append({'id':selected['id'], 'truth':truths[selected['id']]['name'],
                       'phases':selected['phases'], 'nu':selected['noise_variance'], 'N':selected['N'],
                       'confidence_contains_truth':covered, 'risk':risk.tolist(),
                       'certified_arm':j, 'false_certificate':bool(j is not None and risk[j]-risk.min()>CERT_TOL+1e-10),
                       'certified_regret':None if j is None else float(risk[j]-risk.min()),
                       'eb_regret':float(risk[selected['eb_arm']]-risk.min()),
                       'static_power2_regret':float(risk[0]-risk.min())})
        journal(root/'score_journal.jsonl', scores[-1])
        STATE['scored'] = len(scores)
        key = json.dumps([selected['phases'],selected['noise_variance'],truths[selected['id']]['name']])
        if key not in exact:
            v = M@p
            observed_P = np.linalg.pinv(A)@A
            e, _ = gaps([affine(b,A,selected['noise_variance'],np.eye(8)) for b in B], M, v-1e-10, v+1e-10)
            projected, _ = gaps([affine(b,A,selected['noise_variance'],observed_P) for b in B], M, v-1e-10, v+1e-10)
            exact[key] = {'covariance_map_rank':int(np.linalg.matrix_rank(M,tol=1e-9)),
                          'full_intervals':e, 'projected_intervals':projected}
            journal(root/'exact_journal.jsonl', {'key':key, **exact[key]})
            STATE['exact_groups'] = len(exact)
    save(root/'scores.json', scores)
    save(root/'exact_diagnostics.json', exact)
    groups = {}
    for phases,N in itertools.product(P['phase_sets'],P['observational_replicates_N']):
        rows = [r for r in scores if r['phases']==phases and r['N']==N]
        chosen = [r for r in rows if r['certified_arm'] is not None]
        groups[json.dumps([phases,N])] = {'cases':len(rows), 'certified':len(chosen),
                                         'coverage':float(np.mean([r['confidence_contains_truth'] for r in rows])),
                                         'false_certificates':sum(r['false_certificate'] for r in rows),
                                         'eb_regret_mean':float(np.mean([r['eb_regret'] for r in rows])),
                                         'static_regret_mean':float(np.mean([r['static_power2_regret'] for r in rows]))}
    statuses = {status:sum(r['intervals']['status']==status for r in selections)
                for status in sorted({r['intervals']['status'] for r in selections})}
    pairs = [pair for r in selections if r['intervals']['status']=='ok' for pair in r['intervals']['pairs']]
    exact_statuses = [value[target]['status'] for value in exact.values()
                      for target in ['full_intervals','projected_intervals']]
    summary = {'cohort_size':len(scores), 'complete':True, 'groups':groups,
               'certificate_population_regret_tolerance':CERT_TOL,
               'confidence_set_coverage':float(np.mean([r['confidence_contains_truth'] for r in scores])),
               'certified':sum(r['certified_arm'] is not None for r in scores),
               'false_certificates':sum(r['false_certificate'] for r in scores),
               'selection_interval_status_counts':statuses,
               'exact_interval_status_counts':{s:exact_statuses.count(s) for s in sorted(set(exact_statuses))},
               'numerical_checks_pass':all(s in ['ok','empty'] for s in statuses) and all(s=='ok' for s in exact_statuses),
               'max_primal_witness_residual':max([p['max_residual'] for p in pairs],default=0.),
               'max_expanded_duality_gap':max([d['expanded_duality_gap'] for p in pairs for d in p['dualities']],default=0.),
               'LP_calls':LP_CALLS, 'total_wall_seconds':time.time()-began,
               'selection_lp_calls':STATE['selection_lp_calls'], 'selection_lp_seconds':STATE['selection_lp_seconds'],
               'exact_lp_calls':LP_CALLS-STATE['selection_lp_calls'], 'exact_lp_seconds':LP_SECONDS-STATE['selection_lp_seconds'],
               'selection_seconds_sum':sum(r['selection_seconds'] for r in selections),
               'interpretation':'Synthetic conditional-family check only; no same-y adaptive risk, natural-image reliability, method novelty or A-conference readiness claim.'}
    save(root/'summary.json', summary)
    STATE['stage'] = 'complete'
    save(root/'resources.json', {**STATE, 'total_lp_calls':LP_CALLS,'total_lp_seconds':LP_SECONDS,
                                'total_wall_seconds':time.time()-began})
    (root/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists():
        raise SystemExit('Refusing to write into an existing run root')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args.root)
    except Exception as error:
        if args.root.exists():
            save(args.root/'INCOMPLETE.json', {'exception':type(error).__name__, 'message':str(error),
                                             'time_unix':time.time(), 'LP_calls':LP_CALLS,
                                             'LP_seconds':LP_SECONDS, 'state':STATE,
                                             'partial_records':'selection_journal.jsonl, score_journal.jsonl, exact_journal.jsonl'})
        raise
