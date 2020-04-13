from valeo.infrastructure.LogManager import LogManager as lm
# NB: Initializing logger here allows "class loaders of application classes" to benefit from the global initialization

logger = lm().logger(__name__)

from valeo.infrastructure import Const, XY_metadata
import valeo.infrastructure.XY_Loader as XY_Loader

if __name__ == "__main__" :
    # data = DfUtil.loadCsvData([Const.rootData() , "train", "traininginputs.csv"])
    # if data is not None:
    #     data.info()

    mt_train = XY_metadata(Const.rootData(), 'traininginputs.csv', 'trainingoutput.csv',
                           [Const.PROC_TRACEINFO], [Const.PROC_TRACEINFO], Const.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader();
    xy_loader.load_XY_df(mt_train)

    mt_test = XY_metadata(Const.rootData(), 'testinputs.csv', None, None, None, None)

# l = ["data", "train", "traininginputs.csv"]
# print(*l[1:])