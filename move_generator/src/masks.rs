use chess_foundation::Bitboard;

pub const ROOK_MASKS: [Bitboard; 64] = [
    Bitboard(0x101010101017E),
    Bitboard(0x202020202027C),
    Bitboard(0x404040404047A),
    Bitboard(0x8080808080876),
    Bitboard(0x1010101010106E),
    Bitboard(0x2020202020205E),
    Bitboard(0x4040404040403E),
    Bitboard(0x8080808080807E),
    Bitboard(0x1010101017E00),
    Bitboard(0x2020202027C00),
    Bitboard(0x4040404047A00),
    Bitboard(0x8080808087600),
    Bitboard(0x10101010106E00),
    Bitboard(0x20202020205E00),
    Bitboard(0x40404040403E00),
    Bitboard(0x80808080807E00),
    Bitboard(0x10101017E0100),
    Bitboard(0x20202027C0200),
    Bitboard(0x40404047A0400),
    Bitboard(0x8080808760800),
    Bitboard(0x101010106E1000),
    Bitboard(0x202020205E2000),
    Bitboard(0x404040403E4000),
    Bitboard(0x808080807E8000),
    Bitboard(0x101017E010100),
    Bitboard(0x202027C020200),
    Bitboard(0x404047A040400),
    Bitboard(0x8080876080800),
    Bitboard(0x1010106E101000),
    Bitboard(0x2020205E202000),
    Bitboard(0x4040403E404000),
    Bitboard(0x8080807E808000),
    Bitboard(0x1017E01010100),
    Bitboard(0x2027C02020200),
    Bitboard(0x4047A04040400),
    Bitboard(0x8087608080800),
    Bitboard(0x10106E10101000),
    Bitboard(0x20205E20202000),
    Bitboard(0x40403E40404000),
    Bitboard(0x80807E80808000),
    Bitboard(0x17E0101010100),
    Bitboard(0x27C0202020200),
    Bitboard(0x47A0404040400),
    Bitboard(0x8760808080800),
    Bitboard(0x106E1010101000),
    Bitboard(0x205E2020202000),
    Bitboard(0x403E4040404000),
    Bitboard(0x807E8080808000),
    Bitboard(0x7E010101010100),
    Bitboard(0x7C020202020200),
    Bitboard(0x7A040404040400),
    Bitboard(0x76080808080800),
    Bitboard(0x6E101010101000),
    Bitboard(0x5E202020202000),
    Bitboard(0x3E404040404000),
    Bitboard(0x7E808080808000),
    Bitboard(0x7E01010101010100),
    Bitboard(0x7C02020202020200),
    Bitboard(0x7A04040404040400),
    Bitboard(0x7608080808080800),
    Bitboard(0x6E10101010101000),
    Bitboard(0x5E20202020202000),
    Bitboard(0x3E40404040404000),
    Bitboard(0x7E80808080808000),
];
pub const BISHOP_MASKS: [Bitboard; 64] = [
    Bitboard(0x40201008040200),
    Bitboard(0x402010080400),
    Bitboard(0x4020100A00),
    Bitboard(0x40221400),
    Bitboard(0x2442800),
    Bitboard(0x204085000),
    Bitboard(0x20408102000),
    Bitboard(0x2040810204000),
    Bitboard(0x20100804020000),
    Bitboard(0x40201008040000),
    Bitboard(0x4020100A0000),
    Bitboard(0x4022140000),
    Bitboard(0x244280000),
    Bitboard(0x20408500000),
    Bitboard(0x2040810200000),
    Bitboard(0x4081020400000),
    Bitboard(0x10080402000200),
    Bitboard(0x20100804000400),
    Bitboard(0x4020100A000A00),
    Bitboard(0x402214001400),
    Bitboard(0x24428002800),
    Bitboard(0x2040850005000),
    Bitboard(0x4081020002000),
    Bitboard(0x8102040004000),
    Bitboard(0x8040200020400),
    Bitboard(0x10080400040800),
    Bitboard(0x20100A000A1000),
    Bitboard(0x40221400142200),
    Bitboard(0x2442800284400),
    Bitboard(0x4085000500800),
    Bitboard(0x8102000201000),
    Bitboard(0x10204000402000),
    Bitboard(0x4020002040800),
    Bitboard(0x8040004081000),
    Bitboard(0x100A000A102000),
    Bitboard(0x22140014224000),
    Bitboard(0x44280028440200),
    Bitboard(0x8500050080400),
    Bitboard(0x10200020100800),
    Bitboard(0x20400040201000),
    Bitboard(0x2000204081000),
    Bitboard(0x4000408102000),
    Bitboard(0xA000A10204000),
    Bitboard(0x14001422400000),
    Bitboard(0x28002844020000),
    Bitboard(0x50005008040200),
    Bitboard(0x20002010080400),
    Bitboard(0x40004020100800),
    Bitboard(0x20408102000),
    Bitboard(0x40810204000),
    Bitboard(0xA1020400000),
    Bitboard(0x142240000000),
    Bitboard(0x284402000000),
    Bitboard(0x500804020000),
    Bitboard(0x201008040200),
    Bitboard(0x402010080400),
    Bitboard(0x2040810204000),
    Bitboard(0x4081020400000),
    Bitboard(0xA102040000000),
    Bitboard(0x14224000000000),
    Bitboard(0x28440200000000),
    Bitboard(0x50080402000000),
    Bitboard(0x20100804020000),
    Bitboard(0x40201008040200),
];