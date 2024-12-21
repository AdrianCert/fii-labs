inductive BinTree (α : Type) where
| Nil : BinTree α
| Node : α -> BinTree α -> BinTree α -> BinTree α
deriving Repr

def BinTree.mirror {α : Type} : BinTree α → BinTree α
| BinTree.Nil => BinTree.Nil
| BinTree.Node v l r => BinTree.Node v (BinTree.mirror r) (BinTree.mirror l)

theorem BinTree.mirror_reversible {α : Type} (t : BinTree α): BinTree.mirror (BinTree.mirror t) = t := by
induction t with
| Nil => simp [BinTree.mirror]
| Node v l r ihl ihr => simp [BinTree.mirror, ihl, ihr]

-- defining new type
def NatTree := BinTree Nat

def inTree (m : Nat) : BinTree Nat → Bool
| BinTree.Nil => false
| BinTree.Node v l r => (m = v) || inTree m l || inTree m r

def maxInTree : BinTree Nat → Option Nat
| BinTree.Nil => Option.none
| BinTree.Node v l r =>
  match maxInTree l, maxInTree r with
  | Option.none, Option.none => Option.some v
  | Option.none, Option.some rv => Option.some (Nat.max v rv)
  | Option.some lv, Option.some rv => Option.some (Nat.max v (Nat.max lv rv))
  | Option.some lv, Option.none => Option.some (Nat.max v lv)

theorem maxInTree_is_inTree (t : BinTree Nat) :
  match maxInTree t with
  | Option.none => True
  | Option.some n => inTree n t = true := by
  induction t with
  | Nil => trivial
  | Node v l r ihl ihr =>
    -- /-

    unfold maxInTree
    cases maxInTree l with
    | none =>
      cases maxInTree r with
      | none => simp [inTree]
      | some rv =>
        simp [inTree]
        apply Or.inl
        sorry

        --  Nat.max_add_left v
        -- rw [inTree.eq_1]
        -- apply Or.inl

    | some lv =>
      cases maxInTree r with
      | none => sorry
      | some rv =>
        simp [inTree]
        cases Nat.le_total lv rv with
        | inl h =>
          simp [Nat.max_eq_right h]
          sorry
        | inr h =>
          simp [Nat.max_eq_left h]
          apply Or.inr
          sorry
          -- exact ihl
          -- here
          --  inTree
          -- unfold inTree

          -- sorry
    -- -/

    -- simp [maxInTree]
    -- cases maxInTree l with
    -- | none =>
    --   cases maxInTree r with
    --   | none => simp [inTree]
    --   | some rv =>
    --     simp [inTree]
    --     apply Or.inr
    --     apply Or.inr
    --     exact ihr
    -- | some lv =>
    --   cases maxInTree r with
    --   | none =>
    --     simp [inTree]
    --     apply Or.inr
    --     apply Or.inl
    --     exact ihl
    --   | some rv =>
    --     simp [inTree]
    --     cases Nat.le_total lv rv with
    --     | Or.inl h =>
    --       simp [Nat.max_eq_right h]
    --       apply Or.inr
    --       apply Or.inr
    --       exact ihr
    --     | Or.inr h =>
    --       simp [Nat.max_eq_left h]
    --       apply Or.inr
    --       apply Or.inl
    --       exact ihl

/- comment line to test the code (add --)
def testTree : BinTree Nat :=
BinTree.Node 10
  (BinTree.Node 5 BinTree.Nil BinTree.Nil)
  (BinTree.Node 20 BinTree.Nil BinTree.Nil)

#eval inTree 20 testTree

--/
