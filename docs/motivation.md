# Sourcerer: Inferring the Quality of Evidence Behind the Claims

The information age has bought both bounty and pestilence. Today, we are deluged with both correct and incorrect information. If we knew how to tell apart right from wrong claims, there would be no rise in misinformation.

Assessing the truth value of a claim, however, is hard. It needs expert knowledge and careful research. Take for instance, fact check operations, which cull a small number of claims and try to validate those claims. When validating, they have to grapple with which parts of wrong, which parts of right but misleading, which parts are debatable, and which parts are right. All in all it is a noisy and time consuming process to vet a few claims. As the rate of production of information increases, thwarting misinformation by checking all the claims seems implausibly expensive.

Rather than assess the claims directly, we can assess the process. Or, in particular, the residue of one part of the process for making the claim---sources. Except for claims based on private experience, e.g., religious experience, claims are based on sources. We can use the features of these sources to infer credibility. The first the number of sources with to make a claim. All else equal, the more number of people who say the same thing, the greater the chances that the claim is true. None of this is to undercut a common observation: lots of people can be wrong about something. A harder test for veracity if a diverse set of people say the same thing. The third test is checking the credibility of the sources.

Relying on the residue is not a panacea. People can simply lie about the source. We want the source to verify what they have been quoted as saying. And in the era of cheap data, this can be easily enabled. Quotes can be linked to video interviews or automatic transcriptions electronically signed by the interviewee. The same system can be scaled to institutions. The downside is that the system may prove onerous. On the other hand, commonly, the same source is cited by many people so a public repository of verified claims and evidence can mitigate much of the burden.

But will this solve the problem? Likely not. For one, people can still commit sins of omission. For two, they can still draft things in misleading ways. For three, trust in sources may not be tied to correctness. All we have done is built a system for establishing provenance. And establishing the provenance is not enough. Instead, we need a system that incentivizes both correctness and presentation that makes correct interpretation highly likely. It is a high bar. But it is the bar---correct and liable to correctly interpreted.

To help news producers clear the bar, we need to equip them with skills and incentivize these. Incentive could come from ratings of correctness of a random sample of their work or it could come from teaching consumers.

To address the problem, we start by building a simple measure that sheds light on the quality of claims made in the news---the number of sources cited in a news article per 100 words. We also provide a dictionary of sources.

We plan to expand the functionality to include a typology of the sources (government, NGO, people on the street, politician, policy staffer, campaign staff, race, gender) and provide some summary  measures of the diversity of sources.

---

## Where this work now lives

The two halves of the argument above became two projects:

- **The typology of sources** — the last paragraph's plan — is built in
  [`covered`](https://github.com/notnews/covered), which parses `NAME, ROLE:`
  speaker turns and third-party attributions out of cable-news transcripts and
  classifies each source on crossed axes (institutional sector, epistemic role,
  citation mode). Sources are the unit of analysis there.

- **The quality-of-evidence measure** — claims per 100 words, and the evidence
  type backing each claim — stays here in `sourcerer`. Claims are the unit of
  analysis, which is why this half needs a language model and the other half
  does not: whether a claim is supported is not recoverable from typography,
  whereas who is speaking usually is.
