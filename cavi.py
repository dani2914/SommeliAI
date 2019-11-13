import torch
import pyro

class CAVI_PlainLDA(object):

    def __init__(self, model, guide, num_topic, num_vocab, num_txt, num_words_per_txt):

        self.model = model
        self.guide = guide
        self.K = num_topic
        self.V = num_vocab
        self.D = num_txt
        self.W = num_words_per_txt

        #self.model()
        self.guide()

    def step(self, doc_list):

#        # inputs
#        indexed_txt_list = [torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4, 5]), torch.tensor([4, 5, 0, 1, 2])]
#        num_topic = 2
#        num_vocab = 6
#        num_txt = 3
#        num_words_per_txt = [3,4,5]

        doc_list = [doc.to(dtype=torch.int) for doc in doc_list]

        eta_0 = torch.ones(self.K, self.V)
        alpha_0 = torch.ones(self.D, self.K)
#
        eta = pyro.param("eta_q")
        alpha = torch.stack(tuple([pyro.param(f"alpha_q_{d}") for d in torch.arange(self.D)]))

        # print(eta)
        # print(alpha[0])

        phi = [torch.empty(len(doc), self.K) for doc in doc_list]

        # update phi (where phi = beta[z_assignment])
        for d in torch.arange(self.D):
            for i in torch.arange(self.W[d]):
                x_di = doc_list[d][i].to(dtype=torch.long)

                phi_numerator = torch.digamma(eta[:, x_di]) + \
                        torch.digamma(torch.sum(eta, axis=1)) + \
                        torch.digamma(alpha[d, :]) + \
                        torch.digamma(torch.sum(alpha[d]))

                phi[d][i] = phi_numerator / torch.sum(phi_numerator)

        # update alpha
        for d in torch.arange(self.D):
            alpha[d] = alpha_0[d] + torch.sum(phi[d], axis=0)


        # update eta
        for v in torch.arange(self.V, dtype=torch.int):
            v_match = [doc==v for doc in doc_list]


            phi_sub = [phi[d][v_match[d]] for d in torch.arange(d) \
                   if len(phi[d][v_match[d]]) != 0]

            if phi_sub == []:
                eta[:,v] = eta_0[:,v]
            else:
                phi_sub = torch.sum(torch.cat(
                    [phi[d][v_match[d]] for d in torch.arange(d) \
                     if len(phi[d][v_match[d]]) != 0]), axis=0)
                eta[:,v] = eta_0[:,v] + phi_sub

        pyro.clear_param_store()
        pyro.param("eta_q", eta)
        for d in torch.arange(self.D):
            pyro.param(f"alpha_q_{d}", alpha[d])

        # calculate loss
        #loss = pyro.infer.Trace_ELBO().loss(self.model, self.guide)

        #return loss


