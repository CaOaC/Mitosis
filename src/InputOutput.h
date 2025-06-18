#pragma once

#include "var.h"
#include "Particle.h"
#include "Box.h"
#include "Macro.h"

class InputOutput
{
public:
	InputOutput();
	void outTopo(Particle& particles, Box& box, bool clearFile);
	void outAdj(Particle& p);
	void outPara();
	void appendXyzTrajectory(unsigned int dimension, const Particle& particles, bool clearFile);
	void appendKickFrequency(Particle& particles, bool clearFile);
	void appendCenterPosition(const Particle& particles, bool clearFile);
	void outContactMap(Particle& p);
	void outDistanceMap(Particle& p, int cnt);
	void appendCorrelationTrajectory(const Particle& particles, bool clearFile);
	void readTopo(Particle& p);
};