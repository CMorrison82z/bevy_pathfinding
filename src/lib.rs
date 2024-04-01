use bevy_math::*;
use itertools::*;
use ndarray::*;
use num::*;
use pathfinding::{
    directed::dijkstra::dijkstra, prelude::{build_path, dijkstra_all}, *
};
use std::collections::HashMap;

// TODO:
// Improve the `neighbors` methods. Currently prone to out of bounds, and inefficient.

// NOTE:
// Epxlicit types because trait bounds are annoying.
pub mod pair_f32 {
    use super::*;

    pub type Pairf32 = (f32, f32);

    pub fn neighbors(p: &Pairf32) -> Vec<Pairf32> {
        let &(x, y) = p;

        iproduct!(-1..=1, -1..=1)
            .filter(|(i, j)| !(*i == 0 && *j == 0))
            .map(|(i, j)| (x + i as f32, y + j as f32))
            .collect()
    }
}

pub mod pair_i32 {
    use super::*;

    pub type Pairi32 = (i32, i32);

    pub fn neighbors(p: &Pairi32) -> Vec<Pairi32> {
        let &(x, y) = p;

        iproduct!(-1..=1, -1..=1)
            .filter(|(i, j)| !(*i == 0 && *j == 0))
            .map(|(i, j)| (x + i, y + j))
            .collect()
    }
}

pub mod pair_usize {
    use super::*;

    pub type Pairusize = (usize, usize);

    // NOTE:
    // Subtle difference between other neighbors.
    // indices are offset by `-1`, and iterate over a range that is `+1` of (-1, 1) to avoid type
    // casting from int to unsigned (because unsigned cannot be negative)
    pub fn neighbors(p: &Pairusize) -> Vec<Pairusize> {
        let &(x, y) = p;

        iproduct!(0..=2, 0..=2)
            .filter(|(i, j)| !(*i == 1 && *j == 1) && (x + i > 0) && (y + j > 0))
            .map(|(i, j)| (x + i - 1, y + j - 1))
            .collect()
    }
}

use pair_usize::*;

pub use ndarray::Array2;
pub type CostField = Array2<u8>;

#[derive(Debug)]
pub enum CardinalDirection {
    North,
    South,
    West,
    East,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

// TODO:
// Eventually, we will remove spatial dependence (the Vec2), also return any of the 8
// predefined directions (N, NE, E, SE, etc.)
#[derive(Debug)]
pub struct VectorField {
    pub center: Vec2,
    pub field: Array2<Option<Vec2>>,
    pub goal: Pairusize,
}

impl VectorField {
    fn _new(center: Vec2, cost_field: &Array2<u8>, goal: &Pairusize) -> Self {
        let (x_size, y_size) = match cost_field.shape() {
            [x, y] => (*x, *y),
            _ => unreachable!(),
        };

        // NOTE:
        // In order to take distances into account for diagonal versus axial nodes, we square the cost
        // field (note that 255 < 2 ^ 16, so we're safely in range) to preserve the primary factor for
        // calculating the integration field, then also add the distance squared of the nodes in
        // cartesian space to the cost.
        let all: HashMap<Pairusize, (Pairusize, u32)> = dijkstra_all(goal, |&p: &Pairusize| {
            pair_usize::neighbors(&p)
                .iter()
                // .map(|(x, y)| {
                //     println!("mappy mc map face");
                //
                //     if cost_field[(*x, *y)] == 255 {
                //         panic!("impassable");
                //     } else {
                //         panic!("{:?}", cost_field[(*x, *y)]);
                //     }
                //
                //     (*x, *y)
                // })
                .filter(|(x, y)| {
                    (0..x_size).contains(x)
                        && (0..y_size).contains(y)
                        && cost_field[(*x, *y)] != u8::MAX
                })
                .map(|&o_p| {
                    (
                        o_p,
                        (cost_field[o_p] as u32).pow(2)
                            + (o_p.0.abs_diff(p.0) + o_p.1.abs_diff(p.1)) as u32,
                    )
                })
                .collect::<Vec<((usize, usize), u32)>>()
        });

        Self {
            center,
            field: Array2::from_shape_fn((x_size, y_size), |(i, j)| {
                if let Some(((parent_x, parent_y), _)) = all.get(&(i, j)) {
                    // NOTE:
                    // `dijkstra_all` treats the goal as the `start`, and as such, we are traveling in reverse
                    // (i.e, the parent node brings us closer to the `start`, which we are using as the `goal`)
                    Some(
                        (Vec2::new(*parent_x as f32, *parent_y as f32)
                            - Vec2::new(i as f32, j as f32))
                        .normalize(),
                    )
                } else {
                    None
                }
            }),
            goal: goal.clone(), // _transformer: Vec2::new(x_size as f32, y_size as f32) + center
        }
    }
    // TODO:
    // Optimize:
    //  - Only 8 possible directions. No need to normalize everytime (use a preconfigured set of
    //  directions).
    pub fn new(center: Vec2, cost_field: &Array2<u8>, goal: &Pairusize) -> Self {
        Self::_new(center, cost_field, goal)
    }

    // NOTE:
    // Creates a vector field centered at ZERO automatically
    pub fn new_at_zero(cost_field: &Array2<u8>, goal: &Pairusize) -> Self {
        Self::_new(Vec2::ZERO, cost_field, goal)
    }

    pub fn get(&self, &index: &Pairusize) -> Option<Vec2> {
        (self.field.get(index)?).clone()
    }

    pub fn get_from_vec2(&self, v: Vec2) -> Option<Vec2> {
        let f_index = to_field_index(&self.field, Some(&self.center), v)?;

        // NOTE:
        // Yes it's cursed...
        // If there is not vector at a point in the vector field,
        // it corresponds to an "infinite" potential, so we will just
        // push them back
        match self.field.get(f_index).map(|v| v.clone())? {
            Some(v2) => Some(v2),
            None => {
                Some((v - from_field_index(&self.field, Some(&self.center), &f_index)).normalize())
            }
        }
    }
}

pub fn to_field_index<T>(a: &Array2<T>, center: Option<&Vec2>, v: Vec2) -> Option<Pairusize> {
    let (sz_x, sz_y) = match a.shape() {
        [x, y] => (*x, *y),
        _ => unreachable!(),
    };

    let &center = center.unwrap_or(&Vec2::ZERO);

    let field_space_v = v - center + Vec2::new(sz_x as f32 / 2., sz_y as f32 / 2.);

    let pair: Pairusize = (
        num::cast(field_space_v.x.round())?,
        num::cast(field_space_v.y.round())?,
    );
    let (x, y) = pair;

    if x < sz_x && y < sz_y {
        Some(pair)
    } else {
        None
    }
}

pub fn from_field_index<T>(a: &Array2<T>, center: Option<&Vec2>, index: &Pairusize) -> Vec2 {
    let (x, y) = index;
    let (sz_x, sz_y) = match a.shape() {
        [x, y] => (*x, *y),
        _ => unreachable!(),
    };

    let &center = center.unwrap_or(&Vec2::ZERO);

    Vec2::new(*x as f32, *y as f32) + center - Vec2::new(sz_x as f32 / 2., sz_y as f32 / 2.)
}

pub fn dijkstra_in_cost_field(cost_field: &Array2<u8>, center: Option<&Vec2>, start: &Pairusize, goal: &Pairusize) -> Option<(Vec<Vec2>, u32)> {
    let (x_size, y_size) = match cost_field.shape() {
        [x, y] => (*x, *y),
        _ => unreachable!(),
    };

    // NOTE:
    // In order to take distances into account for diagonal versus axial nodes, we square the cost
    // field (note that 255 < 2 ^ 16, so we're safely in range) to preserve the primary factor for
    // calculating the integration field, then also add the distance squared of the nodes in
    // cartesian space to the cost.
    dijkstra(start, |&p: &Pairusize| {
        pair_usize::neighbors(&p)
            .iter()
            .filter(|(x, y)| {
                (0..x_size).contains(x)
                    && (0..y_size).contains(y)
                    && cost_field[(*x, *y)] != u8::MAX
            })
            .map(|&o_p| {
                (
                    o_p,
                    (cost_field[o_p] as u32).pow(2)
                        + (o_p.0.abs_diff(p.0) + o_p.1.abs_diff(p.1)) as u32,
                )
            })
            .collect::<Vec<((usize, usize), u32)>>()
    }, |node| node == goal).map(|(v, c)| (v.iter().map(|pair| from_field_index(cost_field, center, pair)).collect(), c))
}
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use plotters::prelude::*;
//
//     #[test]
//     fn test_add() {
//         let array: Array2<u8> =
//             Array2::from_shape_fn((5, 5), |(i, j)| if i == 2 && j == 2 { 0 } else { 1 });
//
//         let t_index = (1, 2);
//
//         let v_field = vector_field(&array, &(2, 2));
//
//         // Print the array
//         println!("{:?}", array);
//         println!("{:?}", array[t_index]);
//
//         let root = BitMapBackend::new("vector_field.png", (640, 480)).into_drawing_area();
//         root.fill(&WHITE);
//
//         let mut chart = ChartBuilder::on(&root)
//             .caption("Vector Field", ("sans-serif", 50).into_font())
//             .build_cartesian_2d(-5f32..5f32, -5f32..5f32)
//             .unwrap();
//
//         chart.configure_mesh().draw();
//
//         for ((x, y), vec) in v_field.field.indexed_iter() {
//             let base_point = (x as f32, y as f32);
//             let end_point = (base_point.0 + 0.25 * vec.x, base_point.1 + 0.25 * vec.y);
//
//             // Plot each vector as a line
//             chart
//                 .draw_series(LineSeries::new(vec![base_point, end_point], &RED))
//                 .unwrap()
//                 .label(format!("({},{})", x, y))
//                 .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
//         }
//         chart.configure_series_labels().draw();
//     }
// }
