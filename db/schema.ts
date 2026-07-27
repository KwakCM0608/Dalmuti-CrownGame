import {
  index,
  integer,
  primaryKey,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

export const onlineRooms = sqliteTable("online_rooms", {
  code: text("code").primaryKey(),
  stateJson: text("state_json").notNull(),
  revision: integer("revision").notNull().default(0),
  createdAt: integer("created_at").notNull(),
  updatedAt: integer("updated_at").notNull(),
  expiresAt: integer("expires_at").notNull(),
}, (table) => [
  uniqueIndex("online_rooms_code_idx").on(table.code),
]);

export const onlineRoomMembers = sqliteTable("online_room_members", {
  roomCode: text("room_code").notNull().references(() => onlineRooms.code, {
    onDelete: "cascade",
  }),
  playerId: text("player_id").notNull(),
  nickname: text("nickname").notNull(),
  nicknameKey: text("nickname_key").notNull(),
  tokenHash: text("token_hash").notNull(),
  createdAt: integer("created_at").notNull(),
  updatedAt: integer("updated_at").notNull(),
  lastSeenAt: integer("last_seen_at").notNull(),
}, (table) => [
  primaryKey({ columns: [table.roomCode, table.playerId] }),
  uniqueIndex("online_room_members_token_idx").on(table.tokenHash),
  uniqueIndex("online_room_members_nickname_idx").on(
    table.roomCode,
    table.nicknameKey,
  ),
]);

export const onlineRoomChatMessages = sqliteTable(
  "online_room_chat_messages",
  {
    seq: integer("seq").primaryKey({ autoIncrement: true }),
    roomCode: text("room_code")
      .notNull()
      .references(() => onlineRooms.code, { onDelete: "cascade" }),
    messageId: text("message_id").notNull(),
    playerId: text("player_id").notNull(),
    authorName: text("author_name").notNull(),
    body: text("body").notNull(),
    createdAt: integer("created_at").notNull(),
  },
  (table) => [
    uniqueIndex("online_room_chat_message_id_idx").on(
      table.roomCode,
      table.messageId,
    ),
    index("online_room_chat_room_seq_idx").on(table.roomCode, table.seq),
    index("online_room_chat_player_time_idx").on(
      table.roomCode,
      table.playerId,
      table.createdAt,
    ),
  ],
);
