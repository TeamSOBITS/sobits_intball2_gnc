#include "guidance/minco/detail/wrench_envelope.hpp"

#include "sobits_intball2_gnc_cpp/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

WrenchEnvelope loadWrenchEnvelope(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp)
    {
        throw std::runtime_error("cannot open wrench envelope: " + path);
    }
    int rows, cols;
    fp >> rows >> cols;
    if (cols != 6)
    {
        throw std::runtime_error("expected 6 columns (wrench dim)");
    }
    WrenchEnvelope env;
    env.F.resize(rows, 6);
    env.G.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            fp >> env.F(i, j);
        }
        fp >> env.G(i);
    }
    return env;
}

}  // namespace

const WrenchEnvelope &wrenchEnvelope()
{
    static const WrenchEnvelope env = loadWrenchEnvelope(kWrenchEnvelopePath);
    return env;
}

ForceFrame forceFrameFrom(const std::optional<std::vector<double>> &q0)
{
    ForceFrame ff;
    if (q0.has_value())
    {
        if (q0->size() != 4)
        {
            throw std::invalid_argument("q0 must have size 4 ([x, y, z, w])");
        }
        ff.body = true;
        ff.R0t = Quaterniond((*q0)[3], (*q0)[0], (*q0)[1], (*q0)[2]).normalized().toRotationMatrix().transpose();
    }
    return ff;
}

}  // namespace sobits_intball2_gnc::guidance
