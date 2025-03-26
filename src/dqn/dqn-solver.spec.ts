import { Mat } from "recurrent-js";
import { Env } from "../env";
import { DQNOpt } from "./dqn-opt";
import { DQNSolver } from "./dqn-solver";

describe('DQNSolver', () => {
  let solver: DQNSolver;
  let env: Env;
  let opt: DQNOpt;

  beforeEach(() => {
    env = new Env(1, 2, 3, 4);
    spyOn(env, 'get').and.callFake((key) => {
      if (key === 'numberOfStates') return 10;
      if (key === 'numberOfActions') return 4;
      return null;
    });

    opt = new DQNOpt();

    solver = new DQNSolver(env, opt);
  });

  it('should initialize with correct number of states and actions', () => {
    expect(solver.numberOfStates).toBe(10);
    expect(solver.numberOfActions).toBe(4);
  });

  it('should initialize training mode correctly', () => {
    expect(solver.getTrainingMode()).toBeTruthy();
    solver.setTrainingModeTo(false);
    expect(solver.getTrainingMode()).toBeFalsy();
  });

  it('should call reset on initialization', () => {
    spyOn(solver, 'reset');
    spyOn(DQNSolver.prototype, 'reset');
    solver = new DQNSolver(env, opt);
    expect(DQNSolver.prototype.reset).toHaveBeenCalled();
    expect(solver.reset).toHaveBeenCalled();
  });

  it('should reset memory and initialize Net', () => {
    solver.reset();
    expect((solver as any).shortTermMemory.s0).toBeNull();
    expect((solver as any).shortTermMemory.a0).toBeNull();
    expect((solver as any).shortTermMemory.r0).toBeNull();
    expect((solver as any).shortTermMemory.s1).toBeNull();
    expect((solver as any).shortTermMemory.a1).toBeNull();
    expect((solver as any).longTermMemory.length).toBe(0);
  });

  describe('decide', () => {
    it('should choose an action using epsilon-greedy policy', () => {
      spyOn<any>(solver, 'epsilonGreedyActionPolicy').and.returnValue(2);
      const action = solver.decide([1, 2, 3]);
      expect(action).toBe(2);
      expect(solver['epsilonGreedyActionPolicy']).toHaveBeenCalled();
    });

    it('should update state and action memory', () => {
      spyOn<any>(solver, 'epsilonGreedyActionPolicy').and.returnValue(3);
      solver.decide([5, 6, 7]);
      expect(solver['shortTermMemory'].a1).toBe(3);
    });
  });

  describe('learn', () => {
    it('should not learn if reward is null or alpha is zero', () => {
      spyOn(solver as any, 'learnFromSarsaTuple');
      spyOn(solver as any, 'addToReplayMemory');

      solver['shortTermMemory'].r0 = null;
      solver.learn(1);
      expect(solver['learnFromSarsaTuple']).not.toHaveBeenCalled();

      Object.defineProperty(solver, 'alpha', { value: 0 });
      solver.learn(1);
      expect(solver['learnFromSarsaTuple']).not.toHaveBeenCalled();
    });

    it('should call learnFromSarsaTuple and addToReplayMemory when conditions are met', () => {
      spyOn(solver as any, 'learnFromSarsaTuple');
      spyOn(solver as any, 'addToReplayMemory');

      solver['shortTermMemory'].r0 = 1;
      solver.learn(1);

      expect(solver['learnFromSarsaTuple']).toHaveBeenCalled();
      expect(solver['addToReplayMemory']).toHaveBeenCalled();
    });

    it('should clip reward if doRewardClipping is enabled', () => {
      Object.defineProperty(solver, 'doRewardClipping', { value: true });
      Object.defineProperty(solver, 'rewardClamp', { value: 1 });

      const state = new Mat(solver.numberOfStates, 1);
      state.setFrom(Array(solver.numberOfStates).fill(0.5));

      (solver as any).shortTermMemory.s0 = state;
      (solver as any).shortTermMemory.s1 = state;
      (solver as any).shortTermMemory.a0 = 0;
      (solver as any).shortTermMemory.a1 = 0;
      (solver as any).shortTermMemory.r0 = 0.9;

      solver.learn(5);
      expect(solver['shortTermMemory'].r0).toBe(1);

      solver.learn(-10);
      expect(solver['shortTermMemory'].r0).toBe(-1);
    });

  });
});
