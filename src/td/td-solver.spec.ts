import { TDEnv } from "./td-env";
import { TDOpt } from "./td-opt";
import { TDSolver } from "./td-solver";

// Mock Environment
class MockEnv extends TDEnv {
  private actions: number[][];
  constructor(states: number, actionsPerState: number) {
    super(1, 2, 3, 4);
    this.actions = Array.from({ length: states }, () => Array.from({ length: actionsPerState }, (_, i) => i));
  }
  protected stox(s: number): number {
    throw new Error("Method not implemented.");
  }
  protected stoy(s: number): number {
    throw new Error("Method not implemented.");
  }
  protected width: number;
  protected height: number;
  protected numberOfStates: number;
  protected numberOfActions: number;
  get(param: string) {
    if (param === 'numberOfStates') return this.actions.length;
    if (param === 'numerOfActions') return this.actions[0].length;
    return undefined;
  }
  allowedActions(state: number) {
    return this.actions[state];
  }
}

// Mock Options
class MockOpt extends TDOpt {
  private options: Record<string, any>;
  constructor(opts: Record<string, any>) {
    super();
    this.options = opts;
  }
  get(key: string) {
    return this.options[key];
  }
}

describe('TDSolver', () => {
  let solver: TDSolver;
  let env: MockEnv;
  let opt: MockOpt;

  beforeEach(() => {
    env = new MockEnv(10, 4);
    opt = new MockOpt({
      alpha: 0.1,
      epsilon: 0.1,
      gamma: 0.9,
      update: 'qlearn',
      smoothPolicyUpdate: false,
      beta: 0.1,
      lambda: 0.9,
      replacingTraces: false,
      qInitVal: 0,
      numberOfPlanningSteps: 5,
    });

    solver = new TDSolver(env, opt);
  });

  it('should initialize correctly', () => {
    expect(solver).toBeDefined();
  });

  it('should reset and initialize Q values', () => {
    solver.reset();
    expect(solver['Q'].length).toBe(env.get('numberOfStates') * env.get('numerOfActions'));
    // expect(solver['Q'].every((val) => val === 0)).toBeTruthy();
  });

  it('should decide an action using epsilon-greedy policy', () => {
    const state = 0;
    spyOn(Math, 'random').and.returnValue(0.05); // Force exploration
    const action = solver.decide(state);
    expect(action).toBeGreaterThanOrEqual(0);
    expect(action).toBeLessThan(4);
  });

  it('should learn from a reward', () => {
    solver.decide(0);
    solver.learn(10);
    expect(solver['r0']).toEqual(10);
  });

  it('should update r0 on the first learn call', () => {
    solver.learn(5);
    expect(solver['r0']).toBe(5);
  });

  it('should not call learnFromTuple on the first call', () => {
    spyOn((solver as any), 'learnFromTuple');
    solver.learn(5);
    expect((solver as any).learnFromTuple).not.toHaveBeenCalled();
  });

  it('should call learnFromTuple on subsequent calls', () => {
    solver['s0'] = 1;
    solver['a0'] = 2;
    solver['s1'] = 3;
    solver['a1'] = 1;
    solver['r0'] = 10;
    spyOn((solver as any), 'learnFromTuple');
    solver.learn(20);
    expect((solver as any).learnFromTuple).toHaveBeenCalledWith(1, 2, 10, 3, 1, 0.9);
  });

  it('should call updateModel and plan when planning steps are enabled', () => {
    solver['s0'] = 1;
    solver['a0'] = 2;
    solver['s1'] = 3;
    solver['a1'] = 1;
    solver['r0'] = 10;

    spyOn((solver as any), 'updateModel');
    spyOn((solver as any), 'plan');

    solver.learn(20);

    expect((solver as any).updateModel).toHaveBeenCalledWith(1, 2, 10, 3);
    expect((solver as any).plan).toHaveBeenCalled();
  });

  it('should initialize Q values correctly', () => {
    solver.reset();
    const numberOfStates = env.get('numberOfStates');
    const numberOfActions = env.get('numerOfActions');
    expect(solver['Q'].length).toBe(numberOfStates * numberOfActions);
  });

  it('should apply uniform random policy correctly to randomPolicies', () => {
    spyOn(env, 'allowedActions').and.returnValue([0, 2]); // Example allowed actions
    solver.reset();

    const numberOfStates = env.get('numberOfStates');
    const expectedPolicyValue = 1.0 / 2; // Since allowedActions has 2 actions

    expect(solver['randomPolicies'][0 * numberOfStates]).toBeCloseTo(expectedPolicyValue);
    expect(solver['randomPolicies'][2 * numberOfStates]).toBeCloseTo(expectedPolicyValue);
  });


  it('should reset state and action memory variables', () => {
    solver.reset();
    expect(solver['s0']).toBeNull();
    expect(solver['s1']).toBeNull();
    expect(solver['a0']).toBeNull();
    expect(solver['a1']).toBeNull();
    expect(solver['r0']).toBeNull();
  });
});
