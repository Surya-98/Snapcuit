/**********
Copyright 2003 ??.  All rights reserved.
Author: 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "bsim2def.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"

/*
 * B2noise (mode, operation, firstModel, ckt, data, OnDens)
 *    This routine names and evaluates all of the noise sources
 *    associated with MOSFET's.  It starts with the model *firstModel and
 *    traverses all of its insts.  It then proceeds to any other models
 *    on the linked list.  The total output noise density generated by
 *    all of the MOSFET's is summed with the variable "OnDens".
 */


int
B2noise (int mode, int operation, GENmodel *genmodel, CKTcircuit *ckt, 
           Ndata *data, double *OnDens)
{
    NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;

    B2model *firstModel = (B2model *) genmodel;
    B2model *model;
    B2instance *inst;
    double tempOnoise;
    double tempInoise;
    double noizDens[B2NSRCS];
    double lnNdens[B2NSRCS];
    int i;

    /* define the names of the noise sources */

    static char *B2nNames[B2NSRCS] = {       /* Note that we have to keep the order */
	"_rd",              /* noise due to rd */        /* consistent with thestrchr definitions */
	"_rs",              /* noise due to rs */        /* in bsim1defs.h */
	"_id",              /* noise due to id */
	"_1overf",          /* flicker (1/f) noise */
	""                  /* total transistor noise */
    };

    for (model=firstModel; model != NULL; model=B2nextModel(model)) {
	for (inst=B2instances(model); inst != NULL; inst=B2nextInstance(inst)) {
        
	    switch (operation) {

	    case N_OPEN:

		/* see if we have to to produce a summary report */
		/* if so, name all the noise generators */

		if (job->NStpsSm != 0) {
		    switch (mode) {

		    case N_DENS:
			for (i=0; i < B2NSRCS; i++) {
			    NOISE_ADD_OUTVAR(ckt, data, "onoise_%s%s", inst->B2name, B2nNames[i]);
			}
			break;

		    case INT_NOIZ:
			for (i=0; i < B2NSRCS; i++) {
			    NOISE_ADD_OUTVAR(ckt, data, "onoise_total_%s%s", inst->B2name, B2nNames[i]);
			    NOISE_ADD_OUTVAR(ckt, data, "inoise_total_%s%s", inst->B2name, B2nNames[i]);
			}
			break;
		    }
		}
		break;

	    case N_CALC:
		switch (mode) {

		case N_DENS:
		    NevalSrc(&noizDens[B2RDNOIZ],&lnNdens[B2RDNOIZ],
				 ckt,THERMNOISE,inst->B2dNodePrime,inst->B2dNode,
				 inst->B2drainConductance * inst->B2m);

		    NevalSrc(&noizDens[B2RSNOIZ],&lnNdens[B2RSNOIZ],
				 ckt,THERMNOISE,inst->B2sNodePrime,inst->B2sNode,
				 inst->B2sourceConductance * inst->B2m);

		    NevalSrc(&noizDens[B2IDNOIZ],&lnNdens[B2IDNOIZ],
				 ckt,THERMNOISE,inst->B2dNodePrime,inst->B2sNodePrime,
                                 (2.0/3.0 * fabs(inst->B2gm * inst->B2m)));

		    NevalSrc(&noizDens[B2FLNOIZ], NULL, ckt,
				 N_GAIN,inst->B2dNodePrime, inst->B2sNodePrime,
				 (double)0.0);
		    noizDens[B2FLNOIZ] *= model->B2fNcoef * inst->B2m *
				 exp(model->B2fNexp *
				 log(MAX(fabs(inst->B2cd),N_MINLOG))) /
				 (data->freq *
				 (inst->B2w - model->B2deltaW * 1e-6) *
				 (inst->B2l - model->B2deltaL * 1e-6) *
				 model->B2Cox * model->B2Cox);
		    lnNdens[B2FLNOIZ] = 
				 log(MAX(noizDens[B2FLNOIZ],N_MINLOG));

		    noizDens[B2TOTNOIZ] = noizDens[B2RDNOIZ] +
						     noizDens[B2RSNOIZ] +
						     noizDens[B2IDNOIZ] +
						     noizDens[B2FLNOIZ];
		    lnNdens[B2TOTNOIZ] = 
				 log(MAX(noizDens[B2TOTNOIZ], N_MINLOG));

		    *OnDens += noizDens[B2TOTNOIZ];

		    if (data->delFreq == 0.0) { 

			/* if we haven't done any previous integration, we need to */
			/* initialize our "history" variables                      */

			for (i=0; i < B2NSRCS; i++) {
			    inst->B2nVar[LNLSTDENS][i] = lnNdens[i];
			}

			/* clear out our integration variables if it's the first pass */

			if (data->freq == job->NstartFreq) {
			    for (i=0; i < B2NSRCS; i++) {
				inst->B2nVar[OUTNOIZ][i] = 0.0;
				inst->B2nVar[INNOIZ][i] = 0.0;
			    }
			}
		    } else {   /* data->delFreq != 0.0 (we have to integrate) */
			for (i=0; i < B2NSRCS; i++) {
			    if (i != B2TOTNOIZ) {
				tempOnoise = Nintegrate(noizDens[i], lnNdens[i],
				      inst->B2nVar[LNLSTDENS][i], data);
				tempInoise = Nintegrate(noizDens[i] * data->GainSqInv ,
				      lnNdens[i] + data->lnGainInv,
				      inst->B2nVar[LNLSTDENS][i] + data->lnGainInv,
				      data);
				inst->B2nVar[LNLSTDENS][i] = lnNdens[i];
				data->outNoiz += tempOnoise;
				data->inNoise += tempInoise;
				if (job->NStpsSm != 0) {
				    inst->B2nVar[OUTNOIZ][i] += tempOnoise;
				    inst->B2nVar[OUTNOIZ][B2TOTNOIZ] += tempOnoise;
				    inst->B2nVar[INNOIZ][i] += tempInoise;
				    inst->B2nVar[INNOIZ][B2TOTNOIZ] += tempInoise;
                                }
			    }
			}
		    }
		    if (data->prtSummary) {
			for (i=0; i < B2NSRCS; i++) {     /* print a summary report */
			    data->outpVector[data->outNumber++] = noizDens[i];
			}
		    }
		    break;

		case INT_NOIZ:        /* already calculated, just output */
		    if (job->NStpsSm != 0) {
			for (i=0; i < B2NSRCS; i++) {
			    data->outpVector[data->outNumber++] = inst->B2nVar[OUTNOIZ][i];
			    data->outpVector[data->outNumber++] = inst->B2nVar[INNOIZ][i];
			}
		    }    /* if */
		    break;
		}    /* switch (mode) */
		break;

	    case N_CLOSE:
		return (OK);         /* do nothing, the main calling routine will close */
		break;               /* the plots */
	    }    /* switch (operation) */
	}    /* for inst */
    }    /* for model */

return(OK);
}
