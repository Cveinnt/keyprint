"""Separate balanced research profile and exact sampling session.

Reuses explicit source alignment and the verified integer draw/commit lifecycle;
never aliases the paced profile or inherits its detector calibration. No language
translation, response rewriting or response-level retries are performed.
"""
from dataclasses import dataclass
import math
import numpy as np

from balanced_score import allocate, POLICY
from paced_source_session import (
    Config, Profile, TokenSourceSession, SourceRequest, MATCH_BYTES, STARTUP_BYTES,
    digest_spec, integer_distribution, IntegerPrepared, PacedSourceSession,
    SamplingFailure)


def policy_spec():
    return dict(version='keyprint-balanced-source-integer-v1', kernel=POLICY,
                ratio_bounds=['1/2','3/2'], radius='1/2',
                allocation='descending_score_half_mass_proportional_ties',
                excluded_mass='exact_normalized_integer_identity',
                draw='integer_rejection_gcd_reduced_no_float_conversion',
                source_match_bytes=MATCH_BYTES, startup_bytes=STARTUP_BYTES,
                source_alignment='all_overlapping_canonical_suffix_matches',
                canonicalization='ASCII_whitespace_only_no_Unicode_normalization',
                repeated_context='identity', single_distinct_label='identity',
                purpose='explicit_source_request',
                detector='uncalibrated_uniform_layer_bits_with_context_deduplication')


@dataclass(frozen=True,init=False)
class BalancedProfile(Profile):
    eos_ids: frozenset[int]

    def __init__(self,token_bytes,*,tokenizer_identity,eos_ids,config=Config()):
        pieces=tuple(token_bytes);eos=frozenset(eos_ids)
        if not eos or any(type(i) is not int or not 0<=i<len(pieces) or pieces[i] is not None for i in eos):
            raise ValueError('Explicit EOS IDs must map to excluded None pieces')
        identity=tokenizer_identity+':'+digest_spec({'policy':policy_spec(),'eos_ids':sorted(eos)})
        super().__init__(pieces,tokenizer_identity=identity,config=config)
        object.__setattr__(self,'eos_ids',eos)


def partition(base,token_ids,profile,key,context,protected):
    if type(profile) is not BalancedProfile:
        raise TypeError('Explicit balanced research profile required')
    if (len(token_ids)!=len(base) or len(set(token_ids))!=len(token_ids)
            or any(type(i) is not int or not 0<=i<len(profile.classes) for i in token_ids)
            or any(type(i) is not int or not 0<=i<len(profile.classes) for i in protected)):
        raise ValueError('Aligned support and valid protected IDs required')
    initial=integer_distribution(base)
    labels={profile.classes[i] for i in token_ids
            if i not in protected and profile.classes[i] is not None}
    if len(labels)<2:
        return initial,dict(score_layers=0,score_adjustments=0,partitioned=False)
    scores={label:sum(profile.bits(key,context,label)) for label in labels}
    values=[None if i in protected else scores.get(profile.classes[i]) for i in token_ids]
    result=allocate(initial,values,profile.config.layers)
    return result,dict(score_layers=profile.config.layers,score_adjustments=1,partitioned=True)


class BalancedSourceSession(PacedSourceSession):
    """Inherit only exact draw/commit; override policy, preparation and receipt."""
    def __init__(self,profile,key,*,condition,request=SourceRequest()):
        if type(profile) is not BalancedProfile:
            raise TypeError('Explicit balanced research profile required; calibration cannot transfer')
        TokenSourceSession.__init__(self,profile,key,condition=condition,request=request)
        self._drawn=None
        self._balanced_counts=dict(score_layers=0,score_adjustments=0,integer_partitions=0)

    def prepare(self,probabilities):
        self._ready()
        if self._pending is not None:raise RuntimeError('Commit or close pending step first')
        if self._steps>=self.profile.config.max_steps:raise RuntimeError('Response step cap reached')
        if not isinstance(probabilities,np.ndarray) or probabilities.dtype!=np.float64:
            raise TypeError('NumPy float64 probabilities required')
        q=probabilities.copy()
        if q.shape!=(len(self.profile.classes),) or not np.isfinite(q).all() or (q<0).any():
            raise ValueError('Invalid base probability vector')
        ids=tuple(map(int,np.flatnonzero(q>0)));base=tuple(float(q[i]) for i in ids)
        if abs(math.fsum(base)-1)>1e-12:raise ValueError('Invalid base probability mass')
        try:
            mode,protected,occurrences=self._decision(q)
            eligible=self._condition=='marked' and mode!='startup_ordinary' and self._context not in self._used
            if eligible:
                dist,numeric=partition(base,ids,self.profile,self._key,self._context,frozenset(protected))
            else:
                dist=integer_distribution(base)
                numeric=dict(score_layers=0,score_adjustments=0,partitioned=False)
        except BaseException:
            self.close()
            raise
        self._last_decision=dict(mode=mode,protected_token_ids=protected,
                                 alignment_occurrences=occurrences,integer_sampling=True,**numeric)
        self._policy_counts[mode]+=1
        self._protected_candidates+=len(protected);self._alignment_occurrences+=occurrences
        for name in ('score_layers','score_adjustments'):self._balanced_counts[name]+=numeric[name]
        self._balanced_counts['integer_partitions']+=numeric['partitioned']
        self._pending=IntegerPrepared(self._steps,ids,dist)
        self._support=q>0;self._drawn=None
        return self._pending

    def source_receipt(self):
        receipt=TokenSourceSession.source_receipt(self)
        receipt['inherited_source_policy_sha256']=receipt.pop('source_policy_sha256')
        receipt.update(source_policy_sha256=digest_spec(policy_spec()),profile_sha256=self.profile.digest,
                       numeric_counters=dict(self._balanced_counts),integer_sampling=True,
                       detector_calibrated=False,quality_acceptance=False)
        return receipt


def replay_scores(profile,key,token_ids):
    """Native-token-path raw score only; no claimed arbitrary-text tokenizer fit."""
    if type(profile) is not BalancedProfile:
        raise TypeError('Balanced profile required for replay')
    if type(key) is not bytes or len(key)!=32:raise ValueError('Exactly 32 key bytes required')
    if len(token_ids)>profile.config.max_steps:raise ValueError('Path exceeds profile cap')
    context,used,ones,events,eos=(),set(),0,0,False
    for token in token_ids:
        if eos:raise ValueError('Tokens after EOS are not one response')
        label=profile.label(token)
        if label is not None:
            if context not in used:
                ones+=sum(profile.bits(key,context,label));events+=1
            used.add(context);context=(*context,label)[-profile.config.history:]
        eos=token in profile.eos_ids
    return dict(profile_sha256=profile.digest,ones=ones,events=events,
                trials=events*profile.config.layers,token_path_only=True,
                detector_calibrated=False,quality_acceptance=False)
