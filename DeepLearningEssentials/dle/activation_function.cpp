#include "activation_function.hpp"

namespace dle {
namespace activation {

	int step(double x) {
		return x < 0.0 ? -1 : 1;
	}

}
}